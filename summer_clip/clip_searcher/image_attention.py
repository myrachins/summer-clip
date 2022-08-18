import typing as tp
from abc import ABC, abstractmethod

import clip
import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data.dataloader import DataLoader

from summer_clip.clip_model import eval_clip
from summer_clip.clip_adapter import train_adapter
from summer_clip.utils.trainer import run_trainer
from summer_clip.utils.trainer import BaseTrainer


class CacheStrategy(ABC):
    @abstractmethod
    def transform(self, image_features: torch.Tensor, image_outs: torch.Tensor) \
            -> tp.Tuple[torch.Tensor, torch.Tensor]:
        pass


class AllLogitsStrategy(CacheStrategy):
    def transform(self, image_features: torch.Tensor, image_outs: torch.Tensor) \
            -> tp.Tuple[torch.Tensor, torch.Tensor]:
        return image_features, image_outs


class ThresholdStrategy(CacheStrategy):
    def __init__(self, threshold: float, use_softmax: bool = True) -> None:
        super().__init__()
        self.threshold = threshold
        self.use_softmax = use_softmax

    def transform(self, image_features: torch.Tensor, image_outs: torch.Tensor) \
            -> tp.Tuple[torch.Tensor, torch.Tensor]:
        image_outs_transformed = F.softmax(image_outs, dim=1) if self.use_softmax else image_outs
        max_probs, _ = image_outs_transformed.max(dim=1)
        confidence_mask = (max_probs >= self.threshold)
        return image_features[:, confidence_mask], image_outs[confidence_mask]


def compute_accuracy(outputs, target):
    acc1, acc5 = train_adapter.accuracy(outputs, target, topk=(1, 5))

    def transform_acc(acc):
        return 100. * acc / target.shape[0]

    return transform_acc(acc1), transform_acc(acc5)


class SaveImageOuts(BaseTrainer):
    def setup_dataset(self):
        self.dataset = hydra.utils.instantiate(self.cfg.dataset)
        test_labels = [label for _, label in self.dataset]
        self.test_labels = torch.IntTensor(test_labels).to(self.cfg.meta.device)

    def setup_model(self):
        device = torch.device(self.cfg.meta.device)
        clip_model, _ = clip.load(self.cfg.clip.model_name, device, jit=False)
        clip_classes = self.cfg.prompting.classes or self.dataset.classes

        self.test_text_features = eval_clip.zeroshot_classifier(clip_model, clip_classes, self.cfg.prompting.templates).to(device)
        self.test_image_features = torch.load(self.cfg.data.image_features_path).to(device)
        self.test_image_features /= self.test_image_features.norm(dim=0, keepdim=True)

        cache_image_features = torch.load(self.cfg.cache.image_features_path).to(device)
        cache_image_outs = torch.load(self.cfg.cache.image_outs_path).to(device)
        self.logger.log_info(f'original-data-size: {cache_image_outs.shape[0]}')
        cache_strategy: CacheStrategy = hydra.utils.instantiate(self.cfg.cache_strategy)
        self.cache_image_features, self.cache_image_outs = cache_strategy.transform(cache_image_features, cache_image_outs)
        self.cache_image_features /= self.cache_image_features.norm(dim=0, keepdim=True)
        self.logger.log_info(f'cache-size: {self.cache_image_outs.shape[0]}')

    def train_loop(self):
        clip_logits = 100. * self.test_image_features.t() @ self.test_text_features
        eval_top1, eval_top5 = compute_accuracy(clip_logits, self.test_labels)
        self.logger.log_info(f'zero-shot clip: acc@1={eval_top1}, acc@5={eval_top5}')
        beta, alpha = self.cfg.cache.beta, self.cfg.cache.alpha

        affinity = self.test_image_features.t() @ self.cache_image_features
        cache_values = F.softmax(self.cache_image_outs)
        cache_logits = (-1 * beta * (1 - affinity)).exp() @ cache_values

        searcher_logits = clip_logits + cache_logits * alpha
        eval_top1, eval_top5 = compute_accuracy(searcher_logits, self.test_labels)
        self.logger.log_info(f'clip-searcher: acc@1={eval_top1}, acc@5={eval_top5}')


@hydra.main(config_path='../conf', config_name='image_attention', version_base='1.1')
def run(cfg: DictConfig) -> None:
    run_trainer(SaveImageOuts, cfg)


if __name__ == '__main__':
    run()
