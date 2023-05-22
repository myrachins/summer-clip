from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


class CacheValueStrategy(ABC):
    @abstractmethod
    def transform(self, cache_outs: torch.Tensor) -> torch.Tensor:
        pass


class HardCacheStrategy(CacheValueStrategy):
    def transform(self, cache_outs: torch.Tensor) -> torch.Tensor:
        _, labels_ids = cache_outs.max(dim=1)
        cache_outs = F.one_hot(labels_ids, num_classes=cache_outs.shape[1]).half()
        return cache_outs


class SoftmaxCacheStrategy(CacheValueStrategy):
    def __init__(self, clip_scale: float, scale: float) -> None:
        super().__init__()
        self.clip_scale = clip_scale
        self.scale = scale

    def transform(self, cache_outs: torch.Tensor) -> torch.Tensor:
        cache_outs = F.softmax(self.clip_scale * self.scale * cache_outs, dim=1)
        return cache_outs
