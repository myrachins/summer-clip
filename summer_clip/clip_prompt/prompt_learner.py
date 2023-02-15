import math
import typing as tp

import hydra
import torch
from torch import nn
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from transformers import get_scheduler, DataCollatorForLanguageModeling


# class ClipTextEncoder(nn.Module):
#     def __init__(self, clip_model: tp.Any) -> None:
#         super().__init_()
#         self.

#     def forward(self, input_embs, input_ids):


def light_cycle(iterable: tp.Iterable) -> tp.Generator[tp.Any, None, None]:
    while True:
        yield from iterable


class LangevinOptim(Optimizer):
    def __init__(self, params, optim_cfg: DictConfig) -> None:
        self.opt = hydra.utils.instantiate(optim_cfg.base_optim, params=params)
        assert isinstance(self.opt.params, dict), "By now only dicts are supported for the params"
        self.beta_start, self.beta_end = optim_cfg.beta_start, optim_cfg.beta_end
        self._set_beta_init()

    def _set_beta_init(self):
        for group in self.opt.params:
            group['lg'] = self.beta_start

    def step(self, *args, **kwargs):
        res = self.opt.step(*args, **kwargs)
        for group in self.opt.params:
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


class LangevinScheduler(_LRScheduler):
    def __init__(self, **kwargs: tp.Any) -> None:
        self.scheduler = get_scheduler(**kwargs)
        self.opt = kwargs['optimizer']
        assert isinstance(self.opt, LangevinOptim), \
            "Only LangevinOptim is supported as an optimizer"
        num_training_steps = kwargs['num_training_steps']
        self.beta_step = math.pow(self.opt.beta_start / self.opt.beta_end, 1 / num_training_steps)

    def step(self):
        self.scheduler.step()
        for group in self.opt.params:
            group['lg'] /= self.beta_step

    def get_last_beta(self):
        return [group['lg'] for group in self.opt.params]

    def get_last_lr(self):
        return self.scheduler.get_last_lr()

    def get_lr(self):
        return self.scheduler.get_lr()


class FluentPromptModel(nn.Module):
    def __init__(self, model_cfg: DictConfig, clip_embs: nn.Embedding, init_ids: torch.IntTensor) -> None:
        self.model_cfg = model_cfg
        self.clip_embs = clip_embs.weight
        self.prompt_ids = init_ids
        self.prompt_embs = nn.Parameter(self.clip_embs[init_ids].detach().clone(), requires_grad=True)

    def get_prompt_embs(self) -> torch.Tensor:
        return self.prompt_embs

    def get_prompt_ids(self) -> torch.Tensor:
        return self.prompt_ids

    def step(self):
        dists = torch.cdist(
            self.prompt_embs.unsqueeze(0), self.clip_embs.unsqueeze(0),
            **self.model_cfg.cdist_kwargs
        ).squeeze(0)
        self.prompt_ids = dists.argmin(dim=1)
        self.prompt_embs.data = self.clip_embs[self.prompt_ids].detach().clone()


class InitTextPrompter:
    def __init__(self, text: str, max_length: tp.Optional[int] = None) -> None:
        self.text = text
        self.max_length = max_length

    def get_ids(self, tokenizer) -> tp.Any:
        truncation = self.max_length is not None
        return tokenizer(self.text, add_special_tokens=False, truncation=truncation, max_length=self.max_length)


class InitTokensPrompter:
    def __init__(self, tokens: list[str]):
        self.tokens = tokens

    def get_ids(self, tokenizer) -> tp.Any:
        return tokenizer(self.tokens, add_special_tokens=False, is_split_into_words=True)


class InitNumTokensPrompter:
    def __init__(self, token: str, length: int):
        self.token = token
        self.length = length

    def get_ids(self, tokenizer) -> tp.Any:
        tokens = [self.token] * self.length
        return tokenizer(tokens, add_special_tokens=False, is_split_into_words=True)


class LeftPromptCollator:
    def __init__(self, tokenizer, embs) -> None:
        self.tokenizer = tokenizer
        self.embs = embs

        self.bos_id = self.tokenizer.bos_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.lm_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    def get_gpt_input(self, prompt_embs, prompt_ids, input_ids):
        prompt_ids = list(prompt_ids)
        input_ids = [
            [self.bos_id] + prompt_ids + list(i_ids)
            for i_ids in input_ids
        ]
        lm_batch = self.lm_collator(input_ids)
        lm_batch = lm_batch.to(prompt_embs.device)
        input_embs = self.embs(lm_batch['input_ids'])
        input_embs[:, 1:prompt_embs.shape[0] + 1, :] = prompt_embs.unsqueeze(0)
        lm_batch.pop('input_ids')
        lm_batch['input_embs'] = input_embs
        return lm_batch
