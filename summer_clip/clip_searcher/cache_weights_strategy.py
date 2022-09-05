from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


class CacheWeightsStrategy(ABC):
    @abstractmethod
    def transform(self, test_image_features: torch.Tensor, cache_image_features: torch.Tensor) -> torch.Tensor:
        """
        test_image_features: not normalized image features of the test images
        cache_image_features: not normalized image features of the selected cache images
        """
        pass


class CacheWeightsNormStrategy(CacheWeightsStrategy):
    def transform(self, test_image_features: torch.Tensor, cache_image_features: torch.Tensor) -> torch.Tensor:
        test_image_features = test_image_features / test_image_features.norm(dim=0, keepdim=True)
        cache_image_features = cache_image_features / cache_image_features.norm(dim=0, keepdim=True)
        return self.transform_norm(test_image_features, cache_image_features)

    @abstractmethod
    def transform_norm(self, test_image_features: torch.Tensor, cache_image_features: torch.Tensor) -> torch.Tensor:
        pass


class TipAdapterWeightsStrategy(CacheWeightsNormStrategy):
    def __init__(self, beta: float) -> None:
        super().__init__()
        self.beta = beta

    def transform_norm(self, test_image_features: torch.Tensor, cache_image_features: torch.Tensor) -> torch.Tensor:
        cache_weights = test_image_features.t() @ cache_image_features
        cache_weights = (-1 * self.beta * (1 - cache_weights)).exp()
        return cache_weights
