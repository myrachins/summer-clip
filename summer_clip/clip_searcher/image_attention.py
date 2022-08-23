import itertools
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


class IndexedCacheStrategy(CacheStrategy):
    @abstractmethod
    def select(self, image_features: torch.Tensor, image_outs: torch.Tensor) -> torch.Tensor:
        pass

    def transform(self, image_features: torch.Tensor, image_outs: torch.Tensor) \
            -> tp.Tuple[torch.Tensor, torch.Tensor]:
        samples_inds = self.select(image_features, image_outs)
        return image_features[:, samples_inds], image_outs[samples_inds]


class AllLogitsStrategy(IndexedCacheStrategy):
    def select(self, image_features: torch.Tensor, image_outs: torch.Tensor) -> torch.Tensor:
        return torch.arange(image_outs.shape[0])


class ThresholdStrategy(IndexedCacheStrategy):
    def __init__(self, threshold: float, use_softmax: bool = True) -> None:
        super().__init__()
        self.threshold = threshold
        self.use_softmax = use_softmax

    def select(self, image_features: torch.Tensor, image_outs: torch.Tensor) -> torch.Tensor:
        image_outs_transformed = F.softmax(image_outs, dim=1) if self.use_softmax else image_outs
        max_probs, _ = image_outs_transformed.max(dim=1)
        confidence_mask = (max_probs >= self.threshold)
        return confidence_mask.nonzero().squeeze()


class TopKStrategy(IndexedCacheStrategy):
    def __init__(self, topk: int) -> None:
        super().__init__()
        self.topk = topk

    def select(self, image_features: torch.Tensor, image_outs: torch.Tensor) -> torch.Tensor:
        pred_outs, label_preds = image_outs.max(dim=1)
        unique_labels = label_preds.unique()
        print(f'Unique pred labels: {len(unique_labels)}')
        samples_ids = []

        for label in unique_labels:
            label_mask = (label_preds == label)
            label_pred_outs = pred_outs[label_mask]
            topk = min(self.topk, label_pred_outs.shape[0])
            _, top_samples_inds = label_pred_outs.topk(topk)
            samples_ids.append(top_samples_inds)

        return torch.cat(samples_ids)


def compute_accuracy(outputs, target):
    acc1, acc5 = train_adapter.accuracy(outputs, target, topk=(1, 5))

    def transform_acc(acc):
        return 100. * acc / target.shape[0]

    return transform_acc(acc1), transform_acc(acc5)


def make_hard_cache(cache_outs: torch.Tensor) -> torch.Tensor:
    _, labels_ids = cache_outs.max(dim=1)
    return F.one_hot(labels_ids, num_classes=cache_outs.shape[1]).half()


class ImageAttention(BaseTrainer):
    def load_labels(self, dataset):
        labels = [label for _, label in dataset]
        return torch.IntTensor(labels).to(self.cfg.meta.device)

    def setup_dataset(self):
        self.dataset = hydra.utils.instantiate(self.cfg.dataset)
        self.test_labels = self.load_labels(self.dataset)
        self.cache_labels: tp.Optional[torch.Tensor] = None
        if self.cfg.cache.dataset:
            cache_dataset = hydra.utils.instantiate(self.cfg.cache.dataset)
            self.cache_labels = self.load_labels(cache_dataset)

    def load_test_text_features(self, device: torch.device):
        clip_model, _ = clip.load(self.cfg.clip.model_name, device, jit=False)
        clip_classes = self.cfg.prompting.classes or self.dataset.classes

        test_text_features = eval_clip.zeroshot_classifier(clip_model, clip_classes, self.cfg.prompting.templates)
        return test_text_features.to(device)

    def build_cache(self, cache_strategy: CacheStrategy, image_features: torch.Tensor, image_outs: torch.Tensor) \
            -> tp.Tuple[torch.Tensor, torch.Tensor]:
        # __main__.type vs full.package.name.type (types are not the same...)
        from summer_clip.clip_searcher.image_attention import IndexedCacheStrategy
        if not isinstance(cache_strategy, IndexedCacheStrategy):
            return cache_strategy.transform(image_features, image_outs)

        samples_inds = cache_strategy.select(image_features, image_outs)
        cache_image_features, cache_image_outs = image_features[:, samples_inds], image_outs[samples_inds]

        if self.cache_labels is not None:
            cache_labels = self.cache_labels[samples_inds]
            eval_top1, eval_top5 = compute_accuracy(cache_image_outs, cache_labels)
            self.logger.log_info(f'internal cache: acc@1={eval_top1}, acc@5={eval_top5}')

        return cache_image_features, cache_image_outs

    def setup_model(self):
        device = torch.device(self.cfg.meta.device)

        self.test_text_features = self.load_test_text_features(device)
        self.test_image_features = torch.load(self.cfg.data.image_features_path).to(device)
        self.test_image_features /= self.test_image_features.norm(dim=0, keepdim=True)

        cache_image_features = torch.load(self.cfg.cache.image_features_path).to(device)
        cache_image_outs = torch.load(self.cfg.cache.image_outs_path).to(device)
        self.logger.log_info(f'original-data-size: {cache_image_outs.shape[0]}')
        cache_strategy: CacheStrategy = hydra.utils.instantiate(self.cfg.cache_strategy)
        self.cache_image_features, self.cache_image_outs = self.build_cache(cache_strategy, cache_image_features, cache_image_outs)
        self.cache_image_features /= self.cache_image_features.norm(dim=0, keepdim=True)
        self.logger.log_info(f'cache-size: {self.cache_image_outs.shape[0]}')

    def train_loop(self):
        clip_logits = 100. * self.test_image_features.t() @ self.test_text_features
        eval_top1, eval_top5 = compute_accuracy(clip_logits, self.test_labels)
        self.logger.log_info(f'zero-shot clip: acc@1={eval_top1}, acc@5={eval_top5}')

        cache_weights = self.test_image_features.t() @ self.cache_image_features
        # self.logger.log_info(f'cache_weights[0]: {cache_weights[0]}')
        # self.logger.log_info(f'cache_weights[1]: {cache_weights[1]}')
        # cache_values = F.softmax(100. * self.cache_image_outs, dim=1)
        cache_values = make_hard_cache(self.cache_image_outs)
        # self.logger.log_info(f'cache_values[0]: {cache_values[0]}')
        # self.logger.log_info(f'cache_values[1]: {cache_values[1]}')

        for beta, alpha in itertools.product(self.cfg.cache.beta, self.cfg.cache.alpha):
            cache_logits = (-1 * beta * (1 - cache_weights)).exp() @ cache_values
            # self.logger.log_info(f'cache_logits[0]: {cache_logits[0]}')
            # self.logger.log_info(f'cache_logits[1]: {cache_logits[1]}')
            searcher_logits = clip_logits + cache_logits * alpha
            # self.logger.log_info(f'searcher_logits[0]: {searcher_logits[0]}')
            # self.logger.log_info(f'searcher_logits[1]: {searcher_logits[1]}')
            eval_top1, eval_top5 = compute_accuracy(searcher_logits, self.test_labels)
            self.logger.log_info(f'clip-searcher ({beta=}, {alpha=}): acc@1={eval_top1}, acc@5={eval_top5}')


@hydra.main(config_path='../conf', config_name='image_attention', version_base='1.1')
def run(cfg: DictConfig) -> None:
    run_trainer(ImageAttention, cfg)


if __name__ == '__main__':
    run()
