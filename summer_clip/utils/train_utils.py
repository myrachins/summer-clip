import typing as tp
from abc import ABC, abstractmethod

import torch
from torch import nn


def get_grouped_params(named_parameters: tp.Iterable[tuple[str, tp.Any]], weight_decay: float,
                       no_decay: tuple[str, ...] = ("bias", "LayerNorm.weight")):
    params_with_wd, params_without_wd = [], []
    for n, p in named_parameters:
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]


class PartlyTrainedModule(ABC, nn.Module):
    @abstractmethod
    def is_train_param(self, param_name: str) -> bool:
        pass

    def training_parameters(self):
        return (param for _, param in self.named_training_parameters())

    def named_training_parameters(self):
        return ((name, param) for name, param in self.named_parameters() if self.is_train_param(name))

    def training_state_dict(self):
        return {name: param for name, param in self.state_dict().items() if self.is_train_param(name)}


def move_batch(batch: dict[str, tp.Any], device: tp.Any) -> dict[str, tp.Any]:
    def move_value(value: tp.Any):
        if isinstance(value, torch.Tensor):
            value = value.to(device)
        return value
    batch = {name: move_value(val) for name, val in batch.items()}
    return batch
