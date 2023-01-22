import typing as tp
from pathlib import Path
from copy import copy

import torch
import torch.utils
import torch.optim
import clip
import hydra
from torch import nn
from tqdm import tqdm
from accelerate import Accelerator
from datasets.load import load_dataset
from omegaconf import DictConfig, OmegaConf
from torch.utils.data.dataloader import DataLoader
from datasets.arrow_dataset import Dataset
from transformers import DataCollatorForLanguageModeling
from transformers import CLIPTokenizer, AutoModelForCausalLM, get_scheduler

from summer_clip.clip_prompt.gpt import ClipGPT, ClipGPTConfig
from summer_clip.utils.trainer import BaseTrainer, run_trainer


def save_step_model(model: ClipGPT, optimizer: torch.optim.Optimizer, accelerator: Accelerator,
                    epoch_num: int, step: int | str, checkpoints_dir: Path) -> None:
    step_dir = checkpoints_dir / f'epoch_{epoch_num}' / f'step_{step}'
    step_dir.mkdir(parents=True, exist_ok=True)

    def save_data(data, data_name):
        with open(step_dir / f'{data_name}.ckpt', 'wb') as f:
            accelerator.save(data, f)

    save_data(model.training_state_dict(), 'model')
    save_data(optimizer.state_dict(), 'optimizer')


def tokenize_dataset(dataset: Dataset, tokenizer: CLIPTokenizer, max_length: int, text_column: str):
    def tokenization(example):
        texts = ["<|startoftext|>" + text for text in example[text_column]]
        return tokenizer(texts, add_special_tokens=False, truncation=True, max_length=max_length)

    encodings = dataset.map(tokenization, batched=True, remove_columns=dataset.column_names)
    return encodings


# The following code is based on the HuggingFace articles:
# 1) https://huggingface.co/course/chapter7/6
# 2) https://huggingface.co/docs/transformers/main/en/perf_train_gpu_one

def get_grouped_params(model: ClipGPT, weight_decay: float, no_decay: tuple[str, ...] = ("bias", "LayerNorm.weight")):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_training_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]


@torch.no_grad()
def evaluate(model: ClipGPT, accelerator: Accelerator, eval_dataloader: DataLoader) -> tuple[float, float]:
    model.eval()
    losses = []
    for batch in tqdm(eval_dataloader, disable=not accelerator.is_local_main_process):
        with torch.no_grad():
            outputs = model(batch["input_ids"], labels=batch["input_ids"])
        losses.append(accelerator.gather(outputs.loss))
    loss = torch.mean(torch.cat(losses))
    try:
        perplexity = torch.exp(loss).item()
    except OverflowError:
        perplexity = float("inf")
    return loss.item(), perplexity


class ClipGPTTrainer(BaseTrainer):
    def setup_dataset(self):
        self.tokenizer = CLIPTokenizer.from_pretrained(self.cfg.clip.tokenizer_id)

        train_dataset: Dataset = load_dataset(**self.cfg.dataset.train.dataset)  # type: ignore
        self.train_dataset = tokenize_dataset(
            train_dataset, self.tokenizer, self.cfg.dataset.train.max_length, self.cfg.dataset.train.text_column
        )
        val_dataset: Dataset = load_dataset(**self.cfg.dataset.val.dataset)  # type: ignore
        self.val_dataset = tokenize_dataset(
            val_dataset, self.tokenizer, self.cfg.dataset.val.max_length, self.cfg.dataset.val.text_column
        )

    def setup_loaders(self):
        ld_cfg = self.cfg.data_loader
        collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        self.loaders = {
            'train': DataLoader(self.train_dataset, collate_fn=collator, **ld_cfg.train),  # type: ignore
            'val': DataLoader(self.val_dataset, collate_fn=collator, **ld_cfg.val)  # type: ignore
        }

    def set_accelerator(self):
        self.accelerator = Accelerator(**self.cfg.accelerator)
        self.model, self.optimizer, self.loaders['train'], self.loaders['val'] = self.accelerator.prepare(
            self.model, self.optimizer, self.loaders['train'], self.loaders['val']
        )

    def setup_scheduler(self):
        self.set_accelerator()
        self.num_training_steps = self.cfg.train.num_train_epochs * len(self.loaders['train'])
        sch_cfg = self.cfg.scheduler
        self.scheduler = get_scheduler(
            name=sch_cfg.name,
            optimizer=self.optimizer,
            num_warmup_steps=sch_cfg.num_warmup_steps,
            num_training_steps=self.num_training_steps
        )

    def setup_optimizer(self):
        params = get_grouped_params(self.model, weight_decay=self.cfg.optim.weight_decay)
        self.optimizer = torch.optim.AdamW(params, **self.cfg.optim.adamw_kwargs)

    def setup_model(self):
        clip_model, _ = clip.load(self.cfg.clip.model_name, 'cpu', jit=False)
        gpt_model = AutoModelForCausalLM.from_pretrained(self.cfg.gpt.model_id)
        self.model = ClipGPT(ClipGPTConfig(**self.cfg.clip_gpt), clip_model.token_embedding, gpt_model)

    def train_epoch(self, epoch_num, epoch_info):
        train_cfg = self.cfg.train
        self.accelerator.print(f'Running epoch {epoch_num} / {train_cfg.num_train_epochs}...')
        model = self.model.train()
        completed_steps = 0

        for step, batch in tqdm(
            enumerate(self.loaders['train'], start=1), disable=not self.accelerator.is_local_main_process
        ):
            loss = model(**batch).loss
            if step % train_cfg.info_steps == 0:
                self.logger.exp_logger.log({
                    "lr": self.scheduler.get_lr(),
                    "samples": step * batch["input_ids"].shape[0],
                    "steps": completed_steps,
                    "loss/train": loss.item() * train_cfg.gradient_accumulation_steps,
                })
            loss = loss / train_cfg.gradient_accumulation_steps
            self.accelerator.backward(loss)
            if step % train_cfg.gradient_accumulation_steps == 0:
                self.accelerator.clip_grad_norm_(self.model.training_parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                completed_steps += 1

            if step % (train_cfg.eval_steps * train_cfg.gradient_accumulation_steps) == 0:
                self.accelerator.print('Evaluating and saving...')
                eval_loss, perplexity = evaluate(self.model, self.accelerator, self.loaders['val'])
                self.logger.exp_logger.log({
                    "loss/eval": eval_loss, "metrics/perplexity": perplexity
                })
                model.train()
                self.accelerator.wait_for_everyone()
                unwrapped_model = self.accelerator.unwrap_model(model)
                save_step_model(
                    unwrapped_model, self.optimizer, self.accelerator,
                    epoch_num, step, Path(train_cfg.checkpoints_dir)
                )

        return epoch_info

    def save_epoch_model(self, epoch_num):
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        save_step_model(
            unwrapped_model, self.optimizer, self.accelerator,
            epoch_num, 'final', Path(self.cfg.train.checkpoints_dir)
        )


@hydra.main(config_path='../conf', config_name='train_gpt', version_base='1.1')
def run(cfg: DictConfig) -> None:
    run_trainer(ClipGPTTrainer, cfg)


if __name__ == '__main__':
    run()
