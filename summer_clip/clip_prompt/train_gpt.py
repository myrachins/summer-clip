import typing as tp
from pathlib import Path

import torch
import torch.utils
import hydra
from torch import optim
from tqdm import tqdm
from accelerate import Accelerator
from datasets.load import load_dataset, load_from_disk
from omegaconf import DictConfig, OmegaConf
from torch.utils.data.dataloader import DataLoader
from datasets.arrow_dataset import Dataset
from transformers import DataCollatorForLanguageModeling
from transformers import CLIPTokenizer, get_scheduler

from summer_clip.clip_prompt.gpt import ClipGPT, load_model
from summer_clip.utils.trainer import BaseTrainer, run_trainer
from summer_clip.clip_prompt.tokenize_dataset import tokenize_dataset


def save_step_model(model: ClipGPT | None, optimizer: optim.Optimizer | None, scheduler: optim.lr_scheduler._LRScheduler | None,
                    accelerator: Accelerator, epoch_num: int, step: int | str, checkpoints_dir: Path) -> None:
    step_dir = checkpoints_dir / f'epoch_{epoch_num}' / f'step_{step}'
    step_dir.mkdir(parents=True, exist_ok=True)

    def save_data(data, data_name):
        with open(step_dir / f'{data_name}.ckpt', 'wb') as f:
            accelerator.save(data, f)

    if model is not None:
        save_data(model.training_state_dict(), 'model')
        OmegaConf.save(model.cfg, step_dir / 'model_cfg.yaml')
    if optimizer is not None:
        save_data(optimizer.state_dict(), 'optimizer')
    if scheduler is not None:
        save_data(scheduler.state_dict(), 'scheduler')


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
        outputs = model(**batch)
        losses.append(accelerator.gather(outputs.loss))
    loss = torch.mean(torch.stack(losses))
    perp = loss.exp()
    return loss.item(), perp.item()


class WikiFilter:
    def __init__(self, text_column: str) -> None:
        self.text_column = text_column

    def is_valid(self, example: dict[str, tp.Any]) -> bool:
        text = example[self.text_column]
        is_invalid = (text == '' or text.startswith(' =') or text.endswith('= \n'))
        return not is_invalid


class ClipGPTTrainer(BaseTrainer):
    def setup_dataset(self):
        self.tokenizer = CLIPTokenizer.from_pretrained(self.cfg.clip.tokenizer_id)
        dt_cfg = self.cfg.dataset

        train_dataset: Dataset = load_from_disk(**dt_cfg.train.dataset)  # type: ignore
        if dt_cfg.train.subpart is not None:
            train_dataset = train_dataset.shuffle(seed=self.cfg.meta.random_state)
            train_part = int(dt_cfg.train.subpart * len(train_dataset))
            train_dataset = train_dataset.select(range(train_part))
        self.train_dataset = train_dataset
        val_dataset: Dataset = load_dataset(**dt_cfg.val.dataset)  # type: ignore
        val_filter = hydra.utils.instantiate(dt_cfg.val.filter)
        val_dataset = val_dataset.filter(val_filter.is_valid, load_from_cache_file=False)
        self.val_dataset = tokenize_dataset(
            val_dataset, self.tokenizer, dt_cfg.val.max_length, dt_cfg.val.text_column
        )

    def setup_loaders(self):
        ld_cfg = self.cfg.data_loader
        collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        self.loaders = {
            'train': DataLoader(self.train_dataset, collate_fn=collator, **ld_cfg.train),  # type: ignore
            'val': DataLoader(self.val_dataset, collate_fn=collator, **ld_cfg.val)  # type: ignore
        }

    def setup_model(self):
        self.model = load_model(self.cfg.clip_gpt)

    def setup_optimizer(self):
        params = get_grouped_params(self.model, weight_decay=self.cfg.optim.weight_decay)
        self.optimizer = optim.AdamW(params, **self.cfg.optim.adamw_kwargs)

    def setup_scheduler(self):
        sch_cfg = self.cfg.scheduler
        num_training_steps = self.cfg.training.epochs_num * len(self.loaders['train'])
        num_warmup_steps = int(num_training_steps * sch_cfg.warmup_part)
        self.scheduler = get_scheduler(
            name=sch_cfg.name,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

    def setup_accelerator(self):
        self.accelerator = Accelerator(**self.cfg.accelerator)
        self.model, self.optimizer, self.loaders['train'], self.loaders['val'], self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.loaders['train'], self.loaders['val'], self.scheduler
        )

    def setup(self):
        super().setup()
        self.setup_accelerator()

    def train_epoch(self, epoch_num, epoch_info):
        train_cfg = self.cfg.training
        self.accelerator.print(f'Running epoch {epoch_num}/{train_cfg.epochs_num}...')
        model = self.model.train()
        eval_steps = range(
            len(self.loaders['train']), 0,
            -(len(self.loaders['train']) // train_cfg.evals_per_epoch)
        )[:train_cfg.evals_per_epoch]
        completed_steps = 0

        for step, batch in enumerate(
            tqdm(self.loaders['train'], disable=not self.accelerator.is_local_main_process), start=1
        ):
            with self.accelerator.accumulate(model):
                loss = model(**batch).loss
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        self.model.training_parameters(), train_cfg.clip_grad_norm
                    )
                    completed_steps += 1
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            if step % train_cfg.info_steps == 0:
                self.logger.exp_logger.log({
                    "lr": self.scheduler.get_last_lr()[0],
                    "samples": step * batch["input_ids"].shape[0],
                    "steps": completed_steps,
                    "loss/train": loss.item()
                })

            if step in eval_steps:
                self.accelerator.wait_for_everyone()
                self.accelerator.print('\nEvaluating and saving...')
                eval_loss, perplexity = evaluate(self.model, self.accelerator, self.loaders['val'])
                self.logger.exp_logger.log({
                    "loss/eval": eval_loss, "metrics/perplexity": perplexity
                })
                model.train()
                if self.accelerator.is_main_process:
                    unwrapped_model: ClipGPT = self.accelerator.unwrap_model(model)  # type: ignore
                    is_last_step = (step == max(eval_steps))  # if range: max(eval_steps[0], eval_steps[-1])
                    optimizer, scheduler = (self.optimizer, self.scheduler) if is_last_step else (None, None)
                    save_step_model(
                        unwrapped_model, optimizer, scheduler,
                        self.accelerator, epoch_num, step, Path(train_cfg.checkpoints_dir)
                    )

        return epoch_info


@hydra.main(config_path='../conf', config_name='train_gpt', version_base='1.1')
def run(cfg: DictConfig) -> None:
    run_trainer(ClipGPTTrainer, cfg)


if __name__ == '__main__':
    run()
