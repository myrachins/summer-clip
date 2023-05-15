import typing as tp
from abc import ABC, abstractmethod

import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

from summer_clip.clip_searcher.utils import load_labels


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


def select_topk_per_label(image_labels: torch.Tensor, image_logits: torch.Tensor, topk: int) -> torch.Tensor:
    samples_ids = []

    for label in image_labels.unique():
        label_mask = (image_labels == label)
        label_pred_outs = image_logits[label_mask]
        label_topk = min(topk, label_pred_outs.shape[0])
        _, top_samples_inds = label_pred_outs.topk(label_topk)
        global_top_samples_inds = label_mask.nonzero().squeeze(dim=1)[top_samples_inds]
        samples_ids.append(global_top_samples_inds)

    return torch.cat(samples_ids)


class TopKStrategy(IndexedCacheStrategy):
    def __init__(self, topk: int) -> None:
        super().__init__()
        self.topk = topk

    def select(self, image_features: torch.Tensor, image_outs: torch.Tensor) -> torch.Tensor:
        image_logits, label_preds = image_outs.max(dim=1)
        print(f'Unique pred labels: {len(label_preds.unique())}')
        return select_topk_per_label(label_preds, image_logits, self.topk)


class TopKProbStrategy(IndexedCacheStrategy):
    def __init__(self, topk: int, scale: float) -> None:
        super().__init__()
        self.scale = scale
        self.topk_strategy = TopKStrategy(topk)

    def select(self, image_features: torch.Tensor, image_outs: torch.Tensor) -> torch.Tensor:
        image_outs = torch.softmax(image_outs * self.scale, dim=1)
        return self.topk_strategy.select(image_features, image_outs)


class TopKPerGoldStrategy(IndexedCacheStrategy):
    def __init__(self, topk: int, cache_dataset: Dataset) -> None:
        super().__init__()
        self.topk = topk
        self.cache_labels = load_labels(cache_dataset)

    def select(self, image_features: torch.Tensor, image_outs: torch.Tensor) -> torch.Tensor:
        cache_labels = self.cache_labels.to(image_outs.device)
        labels_indexes = cache_labels.long().unsqueeze(dim=0).t()
        image_logits = image_outs.gather(1, labels_indexes).squeeze(dim=1)
        return select_topk_per_label(cache_labels, image_logits, self.topk)


class TopKPerGoldProbStrategy(IndexedCacheStrategy):
    def __init__(self, topk: int, cache_dataset: Dataset, scale: float) -> None:
        super().__init__()
        self.scale = scale
        self.topk_strategy = TopKPerGoldStrategy(topk, cache_dataset)

    def select(self, image_features: torch.Tensor, image_outs: torch.Tensor) -> torch.Tensor:
        image_outs = torch.softmax(image_outs * self.scale, dim=1)
        return self.topk_strategy.select(image_features, image_outs)


class GlobalRandomSampleStrategy(IndexedCacheStrategy):
    def __init__(self, topk: int) -> None:
        super().__init__()
        self.topk = topk

    def select(self, image_features: torch.Tensor, image_outs: torch.Tensor) -> torch.Tensor:
        samples_num = self.topk * image_outs.shape[1]
        samples_num = min(samples_num, image_outs.shape[0])
        samples_ids = np.random.choice(image_outs.shape[0], size=samples_num, replace=False)
        return torch.LongTensor(samples_ids).to(image_outs.device)


def select_k_random_per_label(image_labels: torch.Tensor, k: int) -> torch.Tensor:
    samples_ids = []

    for label in image_labels.unique():
        label_inds = (image_labels == label).nonzero().squeeze()
        label_k = min(k, label_inds.shape[0])
        label_samples_inds_np = np.random.choice(label_inds.shape[0], size=label_k, replace=False)
        label_samples_inds = torch.LongTensor(label_samples_inds_np).to(label_inds.device)
        global_top_samples_inds = label_inds[label_samples_inds]
        samples_ids.append(global_top_samples_inds)

    return torch.cat(samples_ids)


class PerGoldClassRandomSampleStrategy(IndexedCacheStrategy):
    def __init__(self, topk: int, cache_dataset: Dataset) -> None:
        super().__init__()
        self.topk = topk
        self.cache_labels = load_labels(cache_dataset)

    def select(self, image_features: torch.Tensor, image_outs: torch.Tensor) -> torch.Tensor:
        return select_k_random_per_label(self.cache_labels, self.topk)


class PerPredClassRandomSampleStrategy(IndexedCacheStrategy):
    def __init__(self, topk: int) -> None:
        super().__init__()
        self.topk = topk

    def select(self, image_features: torch.Tensor, image_outs: torch.Tensor) -> torch.Tensor:
        _, label_preds = image_outs.max(dim=1)
        unique_labels = label_preds.unique()
        print(f'Unique pred labels: {len(unique_labels)}')
        return select_k_random_per_label(label_preds, self.topk)
