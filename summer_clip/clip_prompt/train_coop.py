import heapq
import typing as tp
from pathlib import Path

import clip
import torch
import torch.utils
import hydra
from munch import Munch
from tqdm import tqdm
from torch import nn
from torch import optim
from omegaconf import DictConfig, OmegaConf
from torch.utils.data.dataloader import DataLoader
from transformers import get_scheduler

from summer_clip.utils.hydra_utils import load_obj
from summer_clip.utils.train_utils import get_grouped_params
from summer_clip.utils.trainer import BaseTrainer, run_trainer
from summer_clip.clip_searcher.utils import load_labels, compute_accuracy
from summer_clip.clip_adapter.train_adapter import NoImageBalancedIndexedDataset
from summer_clip.clip_prompt.prompt_learner import ClipTextEncoder


def set_requires_grad(model: tp.Any, requires_grad: bool) -> None:
    for param in model.parameters():
        param.requires_grad_(requires_grad)


def save_epoch_model(model: tp.Any | None, optimizer: optim.Optimizer | None, scheduler: optim.lr_scheduler._LRScheduler | None,
                     epoch_num: int, checkpoints_dir: Path) -> None:
    epoch_dir = checkpoints_dir / f'epoch_{epoch_num}'
    epoch_dir.mkdir(parents=True, exist_ok=True)

    def save_data(data, data_name):
        with open(epoch_dir / f'{data_name}.ckpt', 'wb') as f:
            torch.save(data, f)

    if model is not None:
        save_data(model.state_dict(), 'model')
    if optimizer is not None:
        save_data(optimizer.state_dict(), 'optimizer')
    if scheduler is not None:
        save_data(scheduler.state_dict(), 'scheduler')


class CoOp(nn.Module):
    def __init__(self, clip_embs: nn.Embedding, prompt_len: int) -> None:
        super().__init__()
        self.prompt_len = prompt_len
        self.prompt_embs = nn.Parameter(torch.randn(prompt_len, clip_embs.weight.shape[1]), requires_grad=True)
        nn.init.normal_(self.prompt_embs, std=0.02)

    def get_prompt_embs(self) -> torch.Tensor:
        return self.prompt_embs

    def get_prompt_ids(self) -> list[int]:
        return [0] * self.prompt_len


class CoOpTrainer(BaseTrainer):
    def setup_dataset(self):
        self.source_dataset = hydra.utils.instantiate(self.cfg.dataset)
        self.dataset = NoImageBalancedIndexedDataset(self.source_dataset, self.cfg.dataset_info.k_shots)

        tokenizer_class = load_obj(self.cfg.tokenizer.path)
        self.tokenizer = tokenizer_class.from_pretrained(self.cfg.tokenizer.name)
        if self.cfg.tokenizer.set_pad_as_eos:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.text_classes = list(self.cfg.prompting.classes or self.source_dataset.classes)
        self.token_classes = self.tokenizer(
            self.text_classes, add_special_tokens=False,
            **self.cfg.tokenizer.tokenize_classes_kwargs
        )['input_ids']

    def setup_loaders(self):
        ld_cfg = self.cfg.data_loader
        self.loaders = {
            'train': DataLoader(self.dataset, **ld_cfg.train),  # type: ignore
        }

    def setup_loss(self):
        self.clip_loss = nn.CrossEntropyLoss()

    def _load_clip_text(self):
        clip_model, _ = clip.load(self.cfg.clip.model_name, 'cpu', jit=False)
        clip_model = clip_model.float()
        text_encoder = ClipTextEncoder(clip_model)
        text_encoder = text_encoder.to(self.device)
        clip_embs = clip_model.token_embedding.to(self.device)
        return text_encoder, clip_embs

    def setup_model(self):
        self.clip_text, clip_embs = self._load_clip_text()
        self.clip_text.eval()
        self.collator = hydra.utils.instantiate(self.cfg.collator, tokenizer=self.tokenizer, embs=clip_embs)
        self.model = CoOp(clip_embs, self.cfg.prompt.len).to(self.device)
        self.image_features = torch.load(self.cfg.clip.image_features_path, map_location='cpu')
        self.image_features = self.image_features.float().t().contiguous()
        self.image_features = self.image_features / self.image_features.norm(dim=1, keepdim=True)

    def compute_text_features(self):
        classes_batch_size = self.cfg.training.classes_batch_size
        prompt_embs = self.model.get_prompt_embs()
        prompt_ids = self.model.get_prompt_ids()
        all_features = []
        for begin_ind in range(0, len(self.token_classes), classes_batch_size):
            end_ind = begin_ind + classes_batch_size
            batch_classes = self.token_classes[begin_ind:end_ind]
            clip_batch = self.collator.get_clip_input(
                prompt_embs=prompt_embs, prompt_ids=prompt_ids,
                input_ids=batch_classes
            )
            text_features = self.clip_text(**clip_batch)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            all_features.append(text_features)
        all_features = torch.cat(all_features, dim=0)
        return all_features

    def compute_clip_metrics(self, labels, indexes):
        text_features = self.compute_text_features()
        image_features = self.image_features[indexes].to(self.device)
        logits = image_features @ text_features.t()
        labels = labels.to(self.device)
        loss = self.clip_loss(logits, labels)
        acc1, acc5 = compute_accuracy(logits, labels)
        return Munch(loss=loss, acc1=acc1, acc5=acc5)

    def setup_optimizer(self):
        params = get_grouped_params(self.model.named_parameters(), weight_decay=self.cfg.optim.weight_decay)
        self.optimizer = optim.AdamW(params, **self.cfg.optim.kwargs)

    def setup_scheduler(self):
        sch_cfg = self.cfg.scheduler
        num_training_steps = (
            (self.cfg.training.epochs_num * len(self.loaders['train']))
            // self.cfg.training.gradient_accumulation_steps  # noqa: W503
        )
        num_warmup_steps = int(num_training_steps * sch_cfg.warmup_part)
        self.scheduler = get_scheduler(
            name=sch_cfg.name,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

    def train_epoch(self, epoch_num, epoch_info):
        train_cfg = self.cfg.training
        print(f'Running epoch {epoch_num}/{train_cfg.epochs_num}...')
        self.model.train()
        completed_steps = 0

        for step, (labels, indexes) in enumerate(tqdm(self.loaders['train']), start=1):
            metrics = self.compute_clip_metrics(labels, indexes)
            loss = metrics.loss
            loss = loss / train_cfg.gradient_accumulation_steps
            loss.backward()

            if step % train_cfg.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                completed_steps += 1

            if step % train_cfg.info_steps == 0:
                self.logger.exp_logger.log({
                    "steps": completed_steps,
                    "loss/train": metrics.loss.item(),
                    "acc/top1": metrics.acc1,
                    "acc/top5": metrics.acc5,
                })

        return epoch_info

    def save_epoch_model(self, epoch_num):
        save_epoch_model(
            self.model, optimizer=None, scheduler=None, epoch_num=epoch_num,
            checkpoints_dir=Path(self.cfg.training.checkpoints_dir)
        )


@hydra.main(config_path='../conf', config_name='train_coop', version_base='1.1')
def run(cfg: DictConfig) -> None:
    run_trainer(CoOpTrainer, cfg)


if __name__ == '__main__':
    run()
