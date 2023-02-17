import heapq
import typing as tp
from pathlib import Path

import clip
import torch
import torch.utils
import hydra
from tqdm import tqdm
from torch import nn
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from torch.utils.data.dataloader import DataLoader
from transformers import CLIPTokenizer

from summer_clip.utils.hydra_utils import load_obj
from summer_clip.clip_prompt.gpt import ClipGPT
from summer_clip.utils.trainer import BaseTrainer, run_trainer
from summer_clip.clip_searcher.utils import load_labels, compute_accuracy
from summer_clip.clip_adapter.train_adapter import NoImageIndexedDataset
from summer_clip.clip_prompt.gen_gpt import load_pretrained_model, load_gpt
from summer_clip.clip_prompt.prompt_learner import GPTEmbed


def save_epoch_model(prompts_items: list[tuple[tp.Any, tp.Any]], tokenizer: CLIPTokenizer,
                     epoch_num: int, checkpoints_dir: Path) -> None:
    epoch_dir = checkpoints_dir / f'epoch_{epoch_num}'
    epoch_dir.mkdir(parents=True, exist_ok=True)

    def get_prompt_tokens(prompt_ids: tp.Any) -> list[str]:
        return [tokenizer.decoder[prompt_id] for prompt_id in prompt_ids]

    records = [
        dict(loss=loss, prompt_ids=p_ids, prompt_tokens=get_prompt_tokens(p_ids))
        for p_ids, loss in prompts_items
    ]
    epoch_cfg = OmegaConf.create(records)
    OmegaConf.save(epoch_cfg, epoch_dir / 'prompts.yaml')


def get_grouped_params(model: ClipGPT, weight_decay: float, no_decay: tuple[str, ...] = ("bias", "LayerNorm.weight")):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]


def set_requires_grad(model: tp.Any, requires_grad: bool) -> None:
    for param in model.parameters():
        param.requires_grad_(requires_grad)


class TopPrompter:
    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self.heap: list[tuple[tp.Any, tp.Any]] = []

    def push(self, prompt_ids, prompt_loss) -> None:
        push_val = (-prompt_loss, prompt_ids)
        max_loss_val = heapq.heappushpop(self.heap, push_val)
        if len(self.heap) < self.max_size:
            heapq.heappush(self.heap, max_loss_val)

    def clear(self) -> None:
        self.heap.clear()

    def items(self) -> list[tuple[tp.Any, tp.Any]]:
        return [(p_ids, -neg_loss) for (neg_loss, p_ids) in sorted(self.heap, reverse=True)]


class PromptTrainer(BaseTrainer):
    def setup_dataset(self):
        self.source_dataset = hydra.utils.instantiate(self.cfg.dataset)
        self.dataset = NoImageIndexedDataset(self.source_dataset)

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

    def _load_gpt_model(self):
        model_cfg = self.cfg.model
        if model_cfg.get('use_clip_gpt', True):
            clip_gpt = load_pretrained_model(
                self.cfg.model.meta_cfg_path, self.cfg.model.state_dict_path,
                map_location=self.accelerator.device
            )
            gpt = GPTEmbed(clip_gpt.gpt)
            embs = clip_gpt.gpt.transformer.wte.emb
        else:
            gpt = load_gpt(self.cfg.model.meta_cfg_path).to(self.accelerator.device)  # type: ignore
            embs = gpt.transformer.wte  # type: ignore
        return gpt, embs

    def setup_model(self):
        self.gpt, clip_embs = self._load_gpt_model()
        set_requires_grad(self.gpt, requires_grad=False)
        set_requires_grad(clip_embs, requires_grad=False)
        init_prompter = hydra.utils.instantiate(self.cfg.init_prompter)
        self.model = hydra.utils.instantiate(
            self.cfg.prompt_model, clip_embs=clip_embs, init_ids=init_prompter.get_ids(self.tokenizer)
        )
        self.collator = hydra.utils.instantiate(self.cfg.collator, tokenizer=self.tokenizer, embs=clip_embs)
        self.text_batcher = load_obj(self.cfg.text_batcher.path)(
            token_classes=self.token_classes, text_classes=self.text_classes, **self.cfg.text_batcher.kwargs
        )
        self.top_prompts = TopPrompter(max_size=self.cfg.training.max_top_prompts)

    def setup_optimizer(self):
        opt_cfg = self.cfg.optim
        params = get_grouped_params(self.model, weight_decay=opt_cfg.weight_decay)
        self.optimizer = load_obj(opt_cfg.optimizer.path)(params=params, **opt_cfg.optimizer.kwargs)

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
        self.gpt, self.model, self.optimizer, self.loaders['train'], self.scheduler = self.accelerator.prepare(
            self.gpt, self.model, self.optimizer, self.loaders['train'], self.scheduler
        )

    def setup(self):
        self.setup_accelerator()
        super().setup()
        self.apply_accelerator()

    def train_epoch(self, epoch_num, epoch_info):
        train_cfg = self.cfg.training
        self.accelerator.print(f'Running epoch {epoch_num}/{train_cfg.epochs_num}...')
        gpt = self.gpt.eval()  # type: ignore
        sum_loss, no_grad_steps, completed_steps = 0., 0, 0

        for step, (labels, indexes) in enumerate(
            tqdm(self.loaders['train'], disable=not self.accelerator.is_local_main_process), start=1
        ):
            batch_classes = self.text_batcher.get_batch_classes(labels)
            prompt_embs = self.model.get_prompt_embs()
            prompt_ids = self.model.get_prompt_ids()
            lm_batch = self.collator.get_gpt_input(
                prompt_embs=prompt_embs, prompt_ids=prompt_ids,
                input_ids=batch_classes
            )
            with self.accelerator.accumulate(gpt):
                loss = gpt(**lm_batch).loss
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(), train_cfg.clip_grad_norm
                    )
                self.optimizer.step()
                self.scheduler.step()
                sum_loss += loss.item()
                no_grad_steps += 1
                if self.accelerator.sync_gradients:
                    avg_loss = sum_loss / no_grad_steps
                    self.top_prompts.push(prompt_ids, avg_loss)
                    self.model.step()
                    completed_steps += 1
                    sum_loss, no_grad_steps = 0., 0
                self.optimizer.zero_grad()

            if step % train_cfg.info_steps == 0:
                self.logger.exp_logger.log({
                    "lr": self.scheduler.get_last_lr()[0],
                    "beta": (self.scheduler.get_last_beta()[0]
                             if hasattr(self.scheduler, 'get_last_beta') else None),
                    "steps": completed_steps,
                    "loss/train": loss.item()
                })

        return epoch_info

    def save_epoch_model(self, epoch_num):
        if self.accelerator.is_main_process:
            save_epoch_model(
                self.top_prompts.items(), self.tokenizer, epoch_num,
                Path(self.cfg.training.checkpoints_dir)
            )
            if self.cfg.training.new_top_prompts_each_epoch:
                self.top_prompts.clear()


@hydra.main(config_path='../conf', config_name='train_prompt', version_base='1.1')
def run(cfg: DictConfig) -> None:
    run_trainer(PromptTrainer, cfg)


if __name__ == '__main__':
    run()
