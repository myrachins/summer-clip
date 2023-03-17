import logging
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
from transformers import DataCollatorForLanguageModeling

from summer_clip.utils.hydra_utils import load_obj
from summer_clip.utils.trainer import BaseTrainer, run_trainer
from summer_clip.clip_adapter.train_adapter import NoImageIndexedDataset, compute_accuracy


torch.no_grad()
def create_text_features(model, tokenizer, classes_tokens, prompts_tokens, device):
    clip_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=77)
    bos_id, eos_id = tokenizer.bos_token_id, tokenizer.eos_token_id
    zeroshot_weights = []
    for class_tokens in tqdm(classes_tokens):
        # TODO: Join it with the collator
        texts = [
            [bos_id] + list(prompt_tokens) + list(class_tokens) + [eos_id]
            for prompt_tokens in prompts_tokens
        ]
        texts = clip_collator(texts)['input_ids'].to(device)
        class_embeddings = model.encode_text(texts)
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        zeroshot_weights.append(class_embedding)
    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights


class PromptEvaluator(BaseTrainer):
    def setup_logger(self):
        pass

    def setup_dataset(self):
        self.source_dataset = hydra.utils.instantiate(self.cfg.dataset)
        self.dataset = NoImageIndexedDataset(self.source_dataset)

        tokenizer_class = load_obj(self.cfg.tokenizer.path)
        self.tokenizer = tokenizer_class.from_pretrained(self.cfg.tokenizer.name)
        self.text_classes = list(self.cfg.prompting.classes or self.source_dataset.classes)
        self.token_classes = self.tokenizer(self.text_classes, add_special_tokens=False)['input_ids']

    def setup_loaders(self):
        ld_cfg = self.cfg.data_loader
        self.loaders = {
            'val': DataLoader(self.dataset, **ld_cfg.val),  # type: ignore
        }

    def setup_prompts(self):
        assert (self.cfg.prompts_ids is None) ^ (self.cfg.prompts_texts is None), "Only one is allowed: text or ids"
        self.token_prompts = self.cfg.prompts_ids
        if self.token_prompts is None:
            self.token_prompts = self.tokenizer(self.cfg.prompts_texts, add_special_tokens=False)['input_ids']

    def setup_model(self):
        self.clip_model, _ = clip.load(self.cfg.clip.model_name, device=self.device, jit=False)
        self.clip_model = self.clip_model.float()
        self.image_features = torch.load(self.cfg.clip.image_features_path, map_location=self.device)
        self.image_features = self.image_features.float()
        self.image_features = self.image_features / self.image_features.norm(dim=0, keepdim=True)
        self.setup_prompts()
        self.text_features = create_text_features(
            self.clip_model, self.tokenizer, self.token_classes, self.token_prompts, self.device
        )

    def train_loop(self):
        top1, top5 = compute_accuracy(self.image_features, self.text_features, self.loaders['val'])
        logging.info(f'acc@1: {top1}')
        logging.info(f'acc@5: {top5}')


@hydra.main(config_path='../conf', config_name='eval_prompt', version_base='1.1')
def run(cfg: DictConfig) -> None:
    run_trainer(PromptEvaluator, cfg)


if __name__ == '__main__':
    run()
