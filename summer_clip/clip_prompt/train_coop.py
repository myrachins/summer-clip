import heapq
import typing as tp
from pathlib import Path

import clip
import torch
import torch.utils
import hydra
import numpy as np
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
from summer_clip.clip_prompt.gen_gpt import load_pretrained_model
from summer_clip.clip_searcher.utils import load_labels, compute_accuracy
from summer_clip.clip_prompt.prompt_learner import ClipTextEncoder, GPTEmbed
from summer_clip.clip_adapter.train_adapter import NoImageBalancedIndexedDataset, NoImageIndexedDataset, accuracy


def set_requires_grad(model: tp.Any, requires_grad: bool) -> None:
    for param in model.parameters():
        param.requires_grad_(requires_grad)


@torch.no_grad()
def compute_accuracy_loader(image_features: torch.Tensor, text_features: torch.Tensor,
                            loader: DataLoader, device: tp.Any) -> tuple[float, float]:
    """
    - image_features: (images_num, emb_dim)
    - text_features: (classes_num, emb_dim)
    """
    text_features = text_features.to(device)
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    top1, top5, n = 0., 0., 0.
    for target, index in tqdm(loader):
        batch_image_features = image_features[index].to(device)
        logits = 100. * batch_image_features @ text_features.t()

        target = target.to(logits.device)
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        top1 += acc1
        top5 += acc5
        n += target.size(0)

    if n <= 0:
        return np.nan, np.nan

    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100

    return top1, top5


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

        self.source_val_dataset = hydra.utils.instantiate(self.cfg.val_dataset)
        self.val_dataset = NoImageIndexedDataset(self.source_val_dataset)

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
            'val': DataLoader(self.val_dataset, **ld_cfg.val),  # type: ignore
        }

    def setup_loss(self):
        self.clip_loss = nn.CrossEntropyLoss()

    def _load_clip_text(self):
        clip_model, _ = clip.load(self.cfg.clip.model_name, 'cpu', jit=False)
        clip_model = clip_model.float()
        text_encoder = ClipTextEncoder(clip_model)
        text_encoder = text_encoder.to(self.device)
        clip_embs = clip_model.token_embedding.to(self.device)
        logit_scale = clip_model.logit_scale.to(self.device).detach()
        return text_encoder, clip_embs, logit_scale

    def _load_gpt_model(self):
        clip_gpt = load_pretrained_model(
            self.cfg.clip_gpt.meta_cfg_path, self.cfg.clip_gpt.state_dict_path,
            map_location=self.device
        )
        gpt = GPTEmbed(clip_gpt.gpt)
        embs = clip_gpt.gpt.transformer.wte.emb
        return gpt, embs

    def _load_image_features(self, image_features_path):
        image_features = torch.load(image_features_path, map_location='cpu')
        image_features = image_features.float().t().contiguous()
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    def prepare_models(self):
        self.gpt.eval()
        self.clip_text.eval()
        set_requires_grad(self.gpt, False)
        set_requires_grad(self.clip_text, False)
        self.logit_scale.requires_grad_(False)

    def setup_model(self):
        self.gpt, _ = self._load_gpt_model()
        self.clip_text, clip_embs, self.logit_scale = self._load_clip_text()
        self.prepare_models()
        self.collator = hydra.utils.instantiate(self.cfg.collator, tokenizer=self.tokenizer, embs=clip_embs)
        self.text_batcher = load_obj(self.cfg.text_batcher.path)(
            token_classes=self.token_classes, text_classes=self.text_classes, **self.cfg.text_batcher.kwargs
        )
        self.lm_loss_transformer = hydra.utils.instantiate(self.cfg.lm_loss)
        self.model = CoOp(clip_embs, self.cfg.prompt.len).to(self.device)
        self.image_features = self._load_image_features(self.cfg.clip.image_features_path)
        self.val_image_features = self._load_image_features(self.cfg.clip.val_image_features_path)

    def compute_text_features(self, prompt_embs, prompt_ids):
        classes_batch_size = self.cfg.training.classes_batch_size
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

    def compute_logits(self, image_features, text_features):
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        return logits

    def compute_clip_metrics(self, labels, indexes, prompt_embs, prompt_ids):
        text_features = self.compute_text_features(prompt_embs, prompt_ids)
        image_features = self.image_features[indexes].to(self.device)
        logits = self.compute_logits(image_features, text_features)
        labels = labels.to(self.device)
        loss = self.clip_loss(logits, labels)
        acc1, acc5 = compute_accuracy(logits, labels)
        return Munch(clip_loss=loss, acc1=acc1, acc5=acc5)

    def compute_lm_loss(self, labels, prompt_embs, prompt_ids):
        batch_classes = self.text_batcher.get_batch_classes(labels)
        lm_batch = self.collator.get_gpt_input(
            prompt_embs=prompt_embs, prompt_ids=prompt_ids,
            input_ids=batch_classes
        )
        lm_out = self.gpt(**lm_batch)
        loss = self.lm_loss_transformer.transform(lm_batch, lm_out)
        return loss

    def compute_full_metrics(self, labels, indexes):
        prompt_embs = self.model.get_prompt_embs()
        prompt_ids = self.model.get_prompt_ids()
        clip_metrics = self.compute_clip_metrics(labels, indexes, prompt_embs, prompt_ids)
        lm_loss = self.compute_lm_loss(labels, prompt_embs, prompt_ids)
        loss = self.cfg.loss.clip * clip_metrics.clip_loss + self.cfg.loss.fluency * lm_loss
        return Munch(loss=loss, lm_loss=lm_loss, clip_loss=clip_metrics.clip_loss, acc1=clip_metrics.acc1, acc5=clip_metrics.acc5)

    def setup_optimizer(self):
        optim_class = load_obj(self.cfg.optim.optim_class)
        params = get_grouped_params(self.model.named_parameters(), weight_decay=self.cfg.optim.weight_decay)
        self.optimizer = optim_class(params, **self.cfg.optim.kwargs)

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
            metrics = self.compute_full_metrics(labels, indexes)
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
                    "loss/clip": metrics.clip_loss.item(),
                    "loss/lm": metrics.lm_loss.item(),
                    "acc/top1": metrics.acc1,
                    "acc/top5": metrics.acc5,
                })

        return epoch_info

    def evaluate_val_model(self):
        self.model.eval()
        text_features = self.compute_text_features(
            self.model.get_prompt_embs(), self.model.get_prompt_ids()
        )
        acc1, acc5 = compute_accuracy_loader(
            self.val_image_features, text_features, self.loaders['val'], self.device
        )
        self.logger.exp_logger.log({
            "eval/acc1": acc1,
            "eval/acc5": acc5,
        })

    def save_epoch_model(self, epoch_num):
        print('Evaluating and saving...')
        self.evaluate_val_model()
        save_epoch_model(
            self.model, optimizer=None, scheduler=None, epoch_num=epoch_num,
            checkpoints_dir=Path(self.cfg.training.checkpoints_dir)
        )


@hydra.main(config_path='../conf', config_name='train_coop', version_base='1.1')
def run(cfg: DictConfig) -> None:
    run_trainer(CoOpTrainer, cfg)


if __name__ == '__main__':
    run()
