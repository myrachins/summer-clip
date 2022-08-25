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
            global_top_samples_inds = label_mask.nonzero().squeeze()[top_samples_inds]
            samples_ids.append(global_top_samples_inds)

        return torch.cat(samples_ids)


class GlobalRandomSampleStrategy(IndexedCacheStrategy):
    def __init__(self, topk: int) -> None:
        super().__init__()
        self.topk = topk

    def select(self, image_features: torch.Tensor, image_outs: torch.Tensor) -> torch.Tensor:
        samples_num = self.topk * image_outs.shape[1]
        samples_ids = np.random.choice(image_outs.shape[0], size=samples_num, replace=False)
        return torch.LongTensor(samples_ids).to(image_outs.device)


class PerGoldClassRandomSampleStrategy(IndexedCacheStrategy):
    def __init__(self, topk: int, cache_dataset: Dataset) -> None:
        super().__init__()
        self.topk = topk
        self.cache_labels = load_labels(cache_dataset)

    def select(self, image_features: torch.Tensor, image_outs: torch.Tensor) -> torch.Tensor:
        samples_ids = []

        for label in self.cache_labels.unique():
            label_inds = (self.cache_labels == label).nonzero().squeeze()
            label_samples_inds_np = np.random.choice(label_inds.shape[0], size=self.topk, replace=False)
            label_samples_inds = torch.LongTensor(label_samples_inds_np).to(label_inds.device)
            global_top_samples_inds = label_inds[label_samples_inds]
            samples_ids.append(global_top_samples_inds)

        return torch.cat(samples_ids)


class PerPredClassRandomSampleStrategy(IndexedCacheStrategy):
    def __init__(self, topk: int) -> None:
        super().__init__()
        self.topk = topk

    def select(self, image_features: torch.Tensor, image_outs: torch.Tensor) -> torch.Tensor:
        _, label_preds = image_outs.max(dim=1)
        unique_labels = label_preds.unique()
        print(f'Unique pred labels: {len(unique_labels)}')
        samples_ids = []

        for label in unique_labels:
            label_inds = (label_preds == label).nonzero().squeeze()
            label_samples_inds_np = np.random.choice(label_inds.shape[0], size=self.topk, replace=False)
            label_samples_inds = torch.LongTensor(label_samples_inds_np).to(label_inds.device)
            global_top_samples_inds = label_inds[label_samples_inds]
            samples_ids.append(global_top_samples_inds)

        return torch.cat(samples_ids)
