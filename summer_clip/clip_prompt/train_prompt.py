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
from omegaconf import DictConfig, OmegaConf
from torch.utils.data.dataloader import DataLoader
from transformers import CLIPTokenizer

from summer_clip.utils.hydra_utils import load_obj
from summer_clip.clip_prompt.gpt import ClipGPT
from summer_clip.utils.trainer import BaseTrainer, run_trainer
from summer_clip.clip_searcher.utils import load_labels, compute_accuracy
from summer_clip.clip_adapter.train_adapter import NoImageBalancedIndexedDataset
from summer_clip.clip_prompt.gen_gpt import load_pretrained_model, load_gpt
from summer_clip.clip_prompt.prompt_learner import GPTEmbed, ClipTextEncoder


def save_step_prompts(prompts_items: list[tuple[tp.Any, tp.Any]], tokenizer: CLIPTokenizer,
                      epoch_num: int, step: int, checkpoints_dir: Path) -> None:
    step_dir = checkpoints_dir / f'epoch_{epoch_num}' / f'step_{step}'
    step_dir.mkdir(parents=True, exist_ok=True)

    def get_prompt_tokens(prompt_ids: tp.Any) -> list[str]:
        return [tokenizer.decoder[prompt_id] for prompt_id in prompt_ids]

    records = [
        dict(loss=loss, prompt_ids=p_ids, prompt_tokens=get_prompt_tokens(p_ids))
        for p_ids, loss in prompts_items
    ]
    epoch_cfg = OmegaConf.create(records)
    OmegaConf.save(epoch_cfg, step_dir / 'prompts.yaml')


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
        return text_encoder

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
        self.clip_text = self._load_clip_text()
        set_requires_grad(self.gpt, requires_grad=False)
        set_requires_grad(clip_embs, requires_grad=False)
        set_requires_grad(self.clip_text, requires_grad=False)
        self.collator = hydra.utils.instantiate(self.cfg.collator, tokenizer=self.tokenizer, embs=clip_embs)
        self.text_batcher = load_obj(self.cfg.text_batcher.path)(
            token_classes=self.token_classes, text_classes=self.text_classes, **self.cfg.text_batcher.kwargs
        )
        init_prompter = hydra.utils.instantiate(self.cfg.init_prompter)
        self.model = hydra.utils.instantiate(
            self.cfg.prompt_model, trainer=self, clip_embs=clip_embs, init_ids=init_prompter.get_ids(self.tokenizer)
        )
        self.image_features = torch.load(self.cfg.clip.image_features_path, map_location='cpu')
        self.image_features = self.image_features.float().t().contiguous()
        self.image_features = self.image_features / self.image_features.norm(dim=1, keepdim=True)
        self.top_prompts = TopPrompter(max_size=self.cfg.training.max_top_prompts)

    def compute_lm_loss(self, labels, prompt_embs, prompt_ids):
        batch_classes = self.text_batcher.get_batch_classes(labels)
        lm_batch = self.collator.get_gpt_input(
            prompt_embs=prompt_embs, prompt_ids=prompt_ids,
            input_ids=batch_classes
        )
        loss = self.gpt(**lm_batch).loss
        return loss

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

    def compute_clip_metrics(self, labels, indexes, prompt_embs, prompt_ids):
        text_features = self.compute_text_features(prompt_embs, prompt_ids)
        image_features = self.image_features[indexes].to(self.device)
        logits = image_features @ text_features.t()
        labels = labels.to(self.device)
        loss = self.clip_loss(logits, labels)
        acc1, acc5 = compute_accuracy(logits, labels)
        return loss, acc1, acc5

    def compute_full_metrics(self, labels, indexes, prompt_embs, prompt_ids):
        clip_loss, acc1, acc5 = self.compute_clip_metrics(labels, indexes, prompt_embs, prompt_ids)
        lm_loss = self.compute_lm_loss(labels, prompt_embs, prompt_ids)
        loss = self.cfg.loss.clip * clip_loss + self.cfg.loss.fluency * lm_loss
        return Munch(loss=loss, clip_loss=clip_loss, lm_loss=lm_loss, acc1=acc1, acc5=acc5)

    def compute_default_metrics(self, labels, indexes):
        prompt_embs = self.model.get_prompt_embs()
        prompt_ids = self.model.get_prompt_ids()
        return self.compute_full_metrics(labels, indexes, prompt_embs, prompt_ids)

    def zero_grad(self):
        self.clip_text.zero_grad()
        self.gpt.zero_grad()
        self.model.zero_grad()

    def set_eval_models(self):
        self.gpt.eval()
        self.clip_text.eval()

    def train_epoch(self, epoch_num, epoch_info):
        train_cfg = self.cfg.training
        print(f'Running epoch {epoch_num}/{train_cfg.epochs_num}...')
        self.set_eval_models()
        avg_loss, completed_steps = 0., 0

        for step, (labels, indexes) in enumerate(tqdm(self.loaders['train']), start=1):
            metrics = self.compute_default_metrics(labels, indexes)
            loss = metrics.loss
            loss = loss / train_cfg.gradient_accumulation_steps
            loss.backward()
            avg_loss += loss.item()

            if step % train_cfg.gradient_accumulation_steps == 0:
                self.top_prompts.push(self.model.get_prompt_ids(), avg_loss)
                self.model.step()
                completed_steps += 1
                avg_loss = 0.

            if step % train_cfg.info_steps == 0:
                self.logger.exp_logger.log({
                    "steps": completed_steps,
                    "loss/train": metrics.loss.item(),
                    "loss/clip": metrics.clip_loss.item(),
                    "loss/lm": metrics.lm_loss.item(),
                    "acc/top1": metrics.acc1,
                    "acc/top5": metrics.acc5,
                })

            if step % train_cfg.save_steps == 0 or step == len(self.loaders['train']):
                save_step_prompts(
                    self.top_prompts.items(), self.tokenizer, epoch_num, step,
                    Path(self.cfg.training.checkpoints_dir)
                )

        return epoch_info

    def save_epoch_model(self, epoch_num):
        if self.cfg.training.new_top_prompts_each_epoch:
            self.top_prompts.clear()


@hydra.main(config_path='../conf', config_name='train_prompt', version_base='1.1')
def run(cfg: DictConfig) -> None:
    run_trainer(PromptTrainer, cfg)


if __name__ == '__main__':
    run()
