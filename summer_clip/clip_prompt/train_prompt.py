import heapq
import typing as tp
from pathlib import Path

import clip
import torch
import torch.utils
import hydra
from tqdm import tqdm
from torch import nn
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
                map_location=self.device
            )
            gpt = GPTEmbed(clip_gpt.gpt)
            embs = clip_gpt.gpt.transformer.wte.emb
        else:
            gpt = load_gpt(self.cfg.model.meta_cfg_path).to(self.device)  # type: ignore
            embs = gpt.transformer.wte  # type: ignore
        return gpt, embs

    def setup_model(self):
        self.gpt, clip_embs = self._load_gpt_model()
        set_requires_grad(self.gpt, requires_grad=False)
        set_requires_grad(clip_embs, requires_grad=False)
        self.collator = hydra.utils.instantiate(self.cfg.collator, tokenizer=self.tokenizer, embs=clip_embs)
        self.text_batcher = load_obj(self.cfg.text_batcher.path)(
            token_classes=self.token_classes, text_classes=self.text_classes, **self.cfg.text_batcher.kwargs
        )
        init_prompter = hydra.utils.instantiate(self.cfg.init_prompter)
        self.model = hydra.utils.instantiate(
            self.cfg.prompt_model, trainer=self, clip_embs=clip_embs, init_ids=init_prompter.get_ids(self.tokenizer)
        )
        self.top_prompts = TopPrompter(max_size=self.cfg.training.max_top_prompts)

    def train_epoch(self, epoch_num, epoch_info):
        train_cfg = self.cfg.training
        print(f'Running epoch {epoch_num}/{train_cfg.epochs_num}...')
        gpt = self.gpt.eval()  # type: ignore
        avg_loss, completed_steps = 0., 0

        for step, (labels, indexes) in enumerate(tqdm(self.loaders['train']), start=1):
            batch_classes = self.text_batcher.get_batch_classes(labels)
            prompt_embs = self.model.get_prompt_embs()
            prompt_ids = self.model.get_prompt_ids()
            lm_batch = self.collator.get_gpt_input(
                prompt_embs=prompt_embs, prompt_ids=prompt_ids,
                input_ids=batch_classes
            )
            loss = gpt(**lm_batch).loss
            loss = loss / train_cfg.gradient_accumulation_steps
            loss.backward()
            avg_loss += loss.item()

            if step % train_cfg.gradient_accumulation_steps == 0:
                self.top_prompts.push(prompt_ids, avg_loss)
                self.model.step()
                completed_steps += 1
                avg_loss = 0.
                gpt.zero_grad()

            if step % train_cfg.info_steps == 0:
                self.logger.exp_logger.log({
                    "steps": completed_steps,
                    "loss/train": loss.item()
                })

        return epoch_info

    def save_epoch_model(self, epoch_num):
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
