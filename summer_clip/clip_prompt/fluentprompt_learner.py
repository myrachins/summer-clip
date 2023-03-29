import math
import typing as tp

import torch
from torch import nn
from omegaconf import DictConfig
from torch.optim import Optimizer
from transformers import get_scheduler
from torch.optim.lr_scheduler import _LRScheduler

from summer_clip.utils.hydra_utils import load_obj


def make_langevin_optim(params: tp.Any, optim_cfg: DictConfig):
    opt_class = load_obj(optim_cfg.base_optim.path)

    class LangevinOptim(opt_class):  # type: ignore
        def __init__(self) -> None:
            super().__init__(params=params, **optim_cfg.base_optim.kwargs)
            self.beta_start, self.beta_end = optim_cfg.beta_start, optim_cfg.beta_end
            self._set_beta_init()

        def _set_beta_init(self):
            for group in self.param_groups:
                group['lg'] = self.beta_start

        def step(self, *args, **kwargs):
            res = super().step(*args, **kwargs)
            for group in self.param_groups:
                params = group['params']
                lr = group['lr']
                lg = group['lg']
                for param in params:
                    if param.grad is None:
                        continue
                    noise = torch.normal(mean=torch.zeros_like(param), std=torch.ones_like(param))
                    noise_coef = math.sqrt(2 * lr * lg)
                    param.data.add_(noise_coef * noise)
            return res

    return LangevinOptim()


class LangevinScheduler(_LRScheduler):
    def __init__(self, **kwargs: tp.Any) -> None:
        self.scheduler = get_scheduler(**kwargs)
        self.opt = kwargs['optimizer']
        assert self.opt.__class__.__name__ == "LangevinOptim", \
            "Only LangevinOptim is supported as an optimizer"
        num_training_steps = kwargs['num_training_steps']
        self.beta_step = math.pow(self.opt.beta_start / self.opt.beta_end, 1 / num_training_steps)

    def step(self):
        self.scheduler.step()
        for group in self.opt.param_groups:
            group['lg'] /= self.beta_step

    def get_last_beta(self):
        return [group['lg'] for group in self.opt.params]

    def get_last_lr(self):
        return self.scheduler.get_last_lr()

    def get_lr(self):
        return self.scheduler.get_lr()


class FluentPromptModel(nn.Module):
    def __init__(self, model_cfg: DictConfig, clip_embs: nn.Embedding, init_ids: list[int]) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.clip_embs = clip_embs.weight.data
        self.prompt_ids = init_ids
        self.prompt_embs = nn.Parameter(self.clip_embs[init_ids].detach().clone(), requires_grad=True)

    def get_prompt_embs(self) -> torch.Tensor:
        return self.prompt_embs

    def get_prompt_ids(self) -> list[int]:
        return self.prompt_ids

    def step(self):
        dists = torch.cdist(
            self.prompt_embs.unsqueeze(0), self.clip_embs.unsqueeze(0),
            **self.model_cfg.cdist_kwargs
        ).squeeze(0)
        prompt_ids = dists.argmin(dim=1)
        self.prompt_embs.data = self.clip_embs[self.prompt_ids].detach().clone()
        self.prompt_ids = prompt_ids.cpu().tolist()
