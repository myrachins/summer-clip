import math
import random
import typing as tp

import hydra
import torch
from torch import nn
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from transformers import DataCollatorForLanguageModeling, get_scheduler

from summer_clip.utils.hydra_utils import load_obj
from summer_clip.clip_prompt.gpt import ClipGPT


class GPTEmbed(nn.Module):
    def __init__(self, gpt: ClipGPT) -> None:
        super().__init__()
        self.gpt = gpt

    def forward(self, inputs_embeds, **kwargs):
        inputs_embeds = self.gpt.transformer.wte.adapter(inputs_embeds)  # type: ignore
        return self.gpt(inputs_embeds=inputs_embeds, **kwargs)


# class ClipTextEncoder(nn.Module):
#     def __init__(self, clip_model: tp.Any) -> None:
#         super().__init_()
#         self.

#     def forward(self, inputs_embeds, input_ids):


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


class InitTextPrompter:
    def __init__(self, text: str, assert_length: tp.Optional[int] = None) -> None:
        self.text = text
        self.assert_length = assert_length

    def get_ids(self, tokenizer) -> tp.Any:
        tokens = tokenizer(self.text, add_special_tokens=False)['input_ids']
        if self.assert_length is not None:
            assert len(tokens) == self.assert_length, "Lens do not match"
        return tokens


class InitTokensPrompter:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens

    def get_ids(self, tokenizer) -> tp.Any:
        return tokenizer(self.tokens, add_special_tokens=False, is_split_into_words=True)['input_ids']


class InitNumTokensPrompter:
    def __init__(self, token: str, length: int) -> None:
        self.token = token
        self.length = length

    def get_ids(self, tokenizer) -> tp.Any:
        tokens = [self.token] * self.length
        return tokenizer(tokens, add_special_tokens=False, is_split_into_words=True)['input_ids']


class InitRandomPrompter:
    def __init__(self, length: int) -> None:
        self.length = length

    def get_ids(self, tokenizer) -> tp.Any:
        special_tokens = (
            'bos_token_id', 'eos_token_id', 'pad_token_id', 'cls_token_id', 'unk_token_id'
        )
        special_tokens_ids = {
            special_token_id for special_token in special_tokens
            if (special_token_id := getattr(tokenizer, special_token, None)) is not None
        }
        tokens_ids = set(range(len(tokenizer))) - special_tokens_ids
        return random.choices(list(tokens_ids), k=self.length)


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
        lm_batch = [dict(input_ids=i_ids, attention_mask=[1] * len(i_ids)) for i_ids in input_ids]
        lm_batch = self.lm_collator(lm_batch)
        lm_batch = {key: val.to(prompt_embs.device) for key, val in lm_batch.items()}
        input_embs = self.embs(lm_batch['input_ids'])
        input_embs[:, 1:prompt_embs.shape[0] + 1, :] = prompt_embs.unsqueeze(0)
        lm_batch.pop('input_ids')
        lm_batch['inputs_embeds'] = input_embs
        return lm_batch


class ImageTextBatcher:
    def __init__(self, token_classes, text_classes):
        self.token_classes = token_classes

    def get_batch_classes(self, batch_labels):
        return [self.token_classes[ind] for ind in batch_labels]


class OneTextBatcher:
    def __init__(self, token_classes, text_classes, class_ind: int) -> None:
        self.token_classes = token_classes
        self.class_ind = class_ind

    def get_batch_classes(self, batch_labels):
        return [self.token_classes[self.class_ind]]


class OneStrTextBatcher(OneTextBatcher):
    def __init__(self, token_classes, text_classes, class_str: str) -> None:
        class_ind = text_classes.index(class_str)
        super().__init__(
            token_classes=token_classes, text_classes=text_classes,
            class_ind=class_ind
        )


class EmptyTextBatcher:
    def __init__(self, token_classes, text_classes):
        pass

    def get_batch_classes(self, batch_labels):
        return [[]]
