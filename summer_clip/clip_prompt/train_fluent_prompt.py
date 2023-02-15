import typing as tp
from pathlib import Path

import clip
import torch
import torch.utils
import hydra
from torch import optim
from tqdm import tqdm
from torch import nn
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from torch.utils.data.dataloader import DataLoader
from datasets.arrow_dataset import Dataset
from transformers import CLIPTokenizer

from summer_clip.utils.hydra_utils import load_obj
from summer_clip.clip_prompt.gpt import ClipGPT, load_model
from summer_clip.utils.trainer import BaseTrainer, run_trainer
from summer_clip.clip_prompt.tokenize_dataset import tokenize_dataset
from summer_clip.clip_searcher.utils import load_labels, compute_accuracy
from summer_clip.clip_adapter.train_adapter import NoImageIndexedDataset
from summer_clip.clip_prompt.prompt_learner import AutoPromptModel, PromptGPT


def save_step_model(model: AutoPromptModel, accelerator: Accelerator, epoch_num: int, checkpoints_dir: Path) -> None:
    step_dir = checkpoints_dir / f'epoch_{epoch_num}'
    step_dir.mkdir(parents=True, exist_ok=True)

    def save_data(data, data_name):
        with open(step_dir / f'{data_name}.ckpt', 'wb') as f:
            accelerator.save(data, f)

    unwrapped_model: ClipGPT = accelerator.unwrap_model(model)  # type: ignore
    save_data(unwrapped_model.training_state_dict(), 'model')
    OmegaConf.save(unwrapped_model.cfg, step_dir / 'model_cfg.yaml')


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


class FluentPromptTrainer(BaseTrainer):
    def setup_dataset(self):
        self.source_dataset = hydra.utils.instantiate(self.cfg.dataset)
        self.dataset = NoImageIndexedDataset(self.source_dataset)
        self.clip_classes = self.cfg.prompting.classes or self.source_dataset.classes

    def setup_loaders(self):
        ld_cfg = self.cfg.data_loader
        self.loaders = {
            'train': DataLoader(self.dataset, **ld_cfg.train),  # type: ignore
        }

    def setup_model(self):
        gpt = load_model(self.cfg.clip_gpt)
        self.gpt = PromptGPT(gpt)
        clip_model, _ = clip.load(self.cfg.clip.model_name, device='cpu', jit=False)
        clip_embs = clip_model.token_embedding
        self.tokenizer = CLIPTokenizer.from_pretrained(self.cfg.clip.tokenizer_id)
        init_prompter = hydra.utils.instantiate(self.cfg.init_prompt)
        self.model = hydra.utils.instantiate(
            self.cfg.prompt_model, clip_embs=clip_embs, init_ids=init_prompter.get_ids(self.tokenizer)
        )

    def setup_optimizer(self):
        params = get_grouped_params(self.model, weight_decay=self.cfg.optim.weight_decay)
        self.optimizer = hydra.utils.instantiate(self.cfg.optim.optimizer, params=params)

    def setup_scheduler(self):
        sch_cfg = self.cfg.scheduler
        num_training_steps = (
            (self.cfg.training.epochs_num * len(self.loaders['train']))
            // self.accelerator.gradient_accumulation_steps  # noqa: W503
        )
        num_warmup_steps = int(num_training_steps * sch_cfg.warmup_part)
        get_scheduler = load_obj(sch_cfg.get_scheduler_fun)
        self.scheduler = get_scheduler(
            name=sch_cfg.name,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

    def setup_accelerator(self):
        self.accelerator = Accelerator(**self.cfg.accelerator)

    def apply_accelerator(self):
        # Val loader should not be sent to 'prepare': problem with the 'end_of_dataloader' state
        # Alternatively could be resolved via restoring states after the evaluation:
        # self.accelerator.gradient_state._set_remainder(remainder)
        # self.accelerator.gradient_state._set_end_of_dataloader(end_of_dataloader)
        self.model, self.optimizer, self.loaders['train'], self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.loaders['train'], self.scheduler
        )

    def setup(self):
        self.setup_accelerator()
        super().setup()
        self.apply_accelerator()

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

            if step in eval_steps and self.accelerator.is_main_process:
                self.accelerator.print('\nEvaluating and saving...')
                eval_loss, perplexity = evaluate(model, self.loaders['val'], self.accelerator.device)
                self.logger.exp_logger.log({
                    "loss/eval": eval_loss, "metrics/perplexity": perplexity
                })
                model.train()
                is_last_step = (step == max(eval_steps))  # if range: max(eval_steps[0], eval_steps[-1])
                optimizer, scheduler = (self.optimizer, self.scheduler) if is_last_step else (None, None)
                save_step_model(
                    model, optimizer, scheduler, self.accelerator,
                    epoch_num, step, Path(train_cfg.checkpoints_dir)
                )

        return epoch_info


@hydra.main(config_path='../conf', config_name='train_fluent_prompt', version_base='1.1')
def run(cfg: DictConfig) -> None:
    run_trainer(FluentPromptTrainer, cfg)


if __name__ == '__main__':
    run()
