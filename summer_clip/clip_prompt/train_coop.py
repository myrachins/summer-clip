from copy import copy
import typing as tp
from pathlib import Path

import clip
import torch
import hydra
import wandb
import torch.utils
import numpy as np
from munch import Munch
from tqdm import tqdm
from torch import nn
from torch import optim
from omegaconf import DictConfig, OmegaConf
from torch.utils.data.dataloader import DataLoader
from transformers import get_scheduler

from summer_clip.utils.hydra_utils import load_obj
from summer_clip.clip_searcher.utils import compute_accuracy
from summer_clip.utils.trainer import BaseTrainer, run_trainer
from summer_clip.clip_prompt.gen_gpt import load_pretrained_model
from summer_clip.utils.train_utils import get_grouped_params, set_requires_grad
from summer_clip.clip_prompt.prompt_learner import ClipTextEncoder, GPTEmbed
from summer_clip.clip_adapter.train_adapter import NoImageBalancedIndexedDataset, NoImageIndexedDataset, accuracy


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


def ids_to_tokens(prompt_ids: list[int], tokenizer: tp.Any) -> list[str]:
    return [tokenizer.decoder[prompt_id] for prompt_id in prompt_ids]


def save_epoch_model(model: tp.Any, optimizer: optim.Optimizer | None, scheduler: optim.lr_scheduler._LRScheduler | None,
                     epoch_num: int, checkpoints_dir: Path) -> None:
    epoch_dir = checkpoints_dir / f'epoch_{epoch_num}'
    epoch_dir.mkdir(parents=True, exist_ok=True)

    def save_data(data, data_name):
        with open(epoch_dir / f'{data_name}.ckpt', 'wb') as f:
            torch.save(data, f)

    save_data(model.state_dict(), 'model')
    if optimizer is not None:
        save_data(optimizer.state_dict(), 'optimizer')
    if scheduler is not None:
        save_data(scheduler.state_dict(), 'scheduler')


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
        self.save_text_classes()

    def setup_loaders(self):
        ld_cfg = self.cfg.data_loader
        self.loaders = {
            'train': DataLoader(self.dataset, **ld_cfg.train),  # type: ignore
            'val': DataLoader(self.val_dataset, **ld_cfg.val),  # type: ignore
        }

    def setup_loss(self):
        self.clip_loss = nn.CrossEntropyLoss()

    def setup_logger(self):
        super().setup_logger()
        self.prompt_records = []

    def save_text_classes(self):
        classes_table = wandb.Table(columns=["ind", "class", "class_tokens"])
        for ind, (class_text, class_ids) in enumerate(zip(self.text_classes, self.token_classes)):
            class_tokens = ids_to_tokens(class_ids, self.tokenizer)
            classes_table.add_data(ind, class_text, class_tokens)
        self.logger.exp_logger.log({"classes_table": classes_table})

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
        self.clip_text, self.clip_embs, self.logit_scale = self._load_clip_text()
        self.prepare_models()
        self.collator = hydra.utils.instantiate(self.cfg.collator, tokenizer=self.tokenizer, embs=self.clip_embs)
        self.text_batcher = load_obj(self.cfg.text_batcher.path)(
            token_classes=self.token_classes, text_classes=self.text_classes, **self.cfg.text_batcher.kwargs
        )
        self.lm_loss_transformer = hydra.utils.instantiate(self.cfg.lm_loss)
        self.model = hydra.utils.instantiate(self.cfg.prompt_model, clip_embs=self.clip_embs).to(self.device)
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

    def make_model_info(self, model_out):
        model_info = copy(model_out)
        for pop_name in ('clip_embs', 'gpt_embs', 'ids'):
            model_info.pop(pop_name)
        return model_info

    def compute_full_metrics(self, labels, indexes):
        model_out = self.model()
        clip_metrics = self.compute_clip_metrics(labels, indexes, model_out.clip_embs, model_out.ids)
        lm_loss = self.compute_lm_loss(labels, model_out.gpt_embs, model_out.ids)
        loss = self.cfg.loss.clip * clip_metrics.clip_loss + self.cfg.loss.fluency * lm_loss
        metrics = Munch(
            loss=loss, lm_loss=lm_loss, clip_loss=clip_metrics.clip_loss,
            acc1=clip_metrics.acc1, acc5=clip_metrics.acc5
        )
        model_info = self.make_model_info(model_out)
        return metrics, model_info

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
            metrics, model_info = self.compute_full_metrics(labels, indexes)
            loss = metrics.loss
            loss = loss / train_cfg.gradient_accumulation_steps
            loss.backward()

            if step % train_cfg.gradient_accumulation_steps == 0:
                model_info |= self.model.step()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                completed_steps += 1

            if step % train_cfg.info_steps == 0:
                main_log = {
                    "steps": completed_steps,
                    "loss/train": metrics.loss.item(),
                    "loss/clip": metrics.clip_loss.item(),
                    "loss/lm": metrics.lm_loss.item(),
                    "acc/top1": metrics.acc1,
                    "acc/top5": metrics.acc5,
                }
                model_info = {f'model_info/{name}': val for name, val in model_info.items()}
                self.logger.exp_logger.log(main_log | model_info)

        return epoch_info

    @torch.no_grad()
    def evaluate_val_model(self, model_out):
        text_features = self.compute_text_features(
            model_out.clip_embs, model_out.ids
        )
        acc1, acc5 = compute_accuracy_loader(
            self.val_image_features, text_features, self.loaders['val'], self.device
        )
        return {
            "eval/acc1": acc1,
            "eval/acc5": acc5,
        }

    def get_embs_from_ids(self, input_ids: list[int]):
        ids = torch.tensor(input_ids, device=self.device)
        embs = self.clip_embs(ids)
        return embs

    @torch.no_grad()
    def evaluate_solo_prompt(self, prompt_embs, prompt_ids):
        lm_batch = self.collator.get_gpt_input(
            prompt_embs=prompt_embs, prompt_ids=prompt_ids,
            input_ids=[[]]
        )
        lm_out = self.gpt(**lm_batch)
        return lm_out.loss.item()

    @torch.no_grad()
    def evaluate_classes_prompt(self, prompt_embs, prompt_ids):
        classes_batch_size = self.cfg.training.classes_batch_size
        res_loss = 0.
        for begin_ind in range(0, len(self.token_classes), classes_batch_size):
            end_ind = begin_ind + classes_batch_size
            batch_classes = self.token_classes[begin_ind:end_ind]
            lm_batch = self.collator.get_gpt_input(
                prompt_embs=prompt_embs, prompt_ids=prompt_ids,
                input_ids=batch_classes
            )
            lm_out = self.gpt(**lm_batch)
            res_loss += lm_out.loss.item() * len(batch_classes)
        res_loss /= len(self.token_classes)
        return res_loss

    @torch.no_grad()
    def evaluate_clip_prompt(self, prompt_embs, prompt_ids):
        text_features = self.compute_text_features(prompt_embs, prompt_ids)
        acc1, acc5 = compute_accuracy_loader(
            self.val_image_features, text_features, self.loaders['val'], self.device
        )
        return acc1, acc5

    @torch.no_grad()
    def evaluate_prompt(self, epoch_num, model_out):
        prompt_embs = self.get_embs_from_ids(model_out.ids)
        prompt_loss = self.evaluate_solo_prompt(prompt_embs, model_out.ids)
        prompt_classes_loss = self.evaluate_classes_prompt(prompt_embs, model_out.ids)
        acc1, acc5 = self.evaluate_clip_prompt(prompt_embs, model_out.ids)
        prompt_tokens = ids_to_tokens(model_out.ids, self.tokenizer)
        prompt_text = self.tokenizer.decode(model_out.ids)
        self.prompt_records.append((
            epoch_num, prompt_loss, prompt_classes_loss,
            acc1, acc5, prompt_text, prompt_tokens
        ))
        prompt_table = wandb.Table(data=self.prompt_records, columns=[
            "epoch", "prompt_loss", "prompt_classes_loss",
            "acc1", "acc5", "prompt", "prompt_tokens"
        ])
        return {
            "prompt_table": prompt_table,
            "prompt/prompt_loss": prompt_loss,
            "prompt/prompt_classes_loss": prompt_classes_loss,
            "prompt/acc1": acc1,
            "prompt/acc5": acc5,
        }

    @torch.no_grad()
    def save_epoch_model(self, epoch_num):
        print('Evaluating and saving...')
        self.model.eval()
        model_out = self.model()
        eval_model = self.evaluate_val_model(model_out)
        eval_prompt = self.evaluate_prompt(epoch_num, model_out)
        self.logger.exp_logger.log(eval_model | eval_prompt)
        save_epoch_model(
            self.model, optimizer=None, scheduler=None,
            epoch_num=epoch_num, checkpoints_dir=Path(self.cfg.training.checkpoints_dir)
        )


@hydra.main(config_path='../conf', config_name='train_coop', version_base='1.1')
def run(cfg: DictConfig) -> None:
    run_trainer(CoOpTrainer, cfg)


if __name__ == '__main__':
    run()
