import typing as tp
from abc import ABC, abstractmethod

import torch
from torch import nn
from torch import Tensor
from munch import Munch
from torch.nn import functional as F
from omegaconf import DictConfig

from summer_clip.utils.hydra_utils import load_obj
from summer_clip.clip_prompt.temp_schedulers import Scheduler
from summer_clip.clip_prompt.prompt_learner import GPTEmbed


def find_nearest(prompt_embs: Tensor, clip_embs: Tensor, p: float) -> Tensor:
    dists = torch.cdist(
        prompt_embs.unsqueeze(0), clip_embs.unsqueeze(0), p=p
    ).squeeze(0)
    prompt_ids = dists.argmin(dim=1)
    return prompt_ids


def straight_through(out_val: Tensor, out_grad: Tensor) -> Tensor:
    out = (out_val - out_grad).detach() + out_grad
    return out


def get_prompt_grads_info(prompt_embs: Tensor, log_dir_name: str = 'prompt_grad_norm') -> Munch:
    prompt_grad = prompt_embs.grad
    if prompt_grad is None:
        return Munch()
    prompt_grad_norms = prompt_grad.norm(dim=-1).detach().cpu()
    grad_info = Munch({
        f"{log_dir_name}/{ind+1}": prompt_grad_norms[ind]
        for ind in range(len(prompt_grad_norms))
    })
    return grad_info


class BasePromptModel(nn.Module):
    def __init__(self, clip_embs: nn.Embedding, prompt_len: int,
                 allowed_tokens: list[int] | None = None, **kwargs: tp.Any) -> None:
        super().__init__()
        self.prompt_len = prompt_len
        self.clip_embs = clip_embs.weight.data  # we are not training this one
        self.allowed_tokens = allowed_tokens

        if allowed_tokens is not None:
            self.clip_embs = self.clip_embs[allowed_tokens]
            self.forward = self.wrap_forward(self.forward, allowed_tokens)  # type: ignore

    def wrap_forward(self, real_forward, allowed_tokens):
        """Is used to transform ids back to global"""
        def wrap(*args, **kwargs):
            out = real_forward(*args, **kwargs)
            out['ids'] = [allowed_tokens[token_id] for token_id in out['ids']]
            return out
        return wrap

    def step(self) -> Munch:
        return Munch()


class CoOp(BasePromptModel):
    def __init__(self, dist_p: float, **kwargs: tp.Any) -> None:
        super().__init__(**kwargs)
        self.dist_p = dist_p
        self.prompt_embs = nn.Parameter(torch.randn(self.prompt_len, self.clip_embs.shape[1]), requires_grad=True)
        nn.init.normal_(self.prompt_embs, std=0.02)

    def forward(self):
        out = Munch(
            clip_embs=self.prompt_embs, gpt_embs=self.prompt_embs,
            ids=self.get_prompt_ids()
        )
        return out

    def get_prompt_ids(self) -> list[int]:
        if self.training:
            prompt_ids = [0] * self.prompt_len
        else:
            prompt_ids = find_nearest(
                self.prompt_embs, self.clip_embs, self.dist_p
            ).cpu().tolist()
        return prompt_ids

    def step(self):
        return get_prompt_grads_info(self.prompt_embs)


class VQVAE1(BasePromptModel):
    def __init__(self, dist_p: float, **kwargs: tp.Any) -> None:
        super().__init__(**kwargs)
        self.dist_p = dist_p
        self.prompt_embs = nn.Parameter(torch.randn(self.prompt_len, self.clip_embs.shape[1]), requires_grad=True)
        nn.init.normal_(self.prompt_embs, std=0.02)

    def forward(self):
        prompt_ids = find_nearest(
            self.prompt_embs, self.clip_embs, self.dist_p
        )
        vocab_embs = self.clip_embs[prompt_ids, :]
        out_embs = straight_through(vocab_embs, self.prompt_embs)
        out = Munch(
            clip_embs=out_embs, gpt_embs=out_embs, ids=prompt_ids.cpu().tolist()
        )
        return out


class VQVAE2(VQVAE1):
    def forward(self):
        out = super().forward()
        out.clip_embs = self.prompt_embs
        return out


class GumbelBase(ABC, BasePromptModel):
    def __init__(self, temp_scheduler: Scheduler, **kwargs: tp.Any) -> None:
        super().__init__(**kwargs)
        self.temp_scheduler = temp_scheduler
        self.logits_log_temperature = torch.tensor(1 / 100).log()  # no training
        self.register_buffer('temperature', torch.tensor(self.temp_scheduler.get_val()))

    @abstractmethod
    def get_prompt_logits(self) -> Tensor:
        pass

    def get_temperature(self):
        if self.training:
            self.temperature = torch.tensor(self.temp_scheduler.get_val())
            self.temp_scheduler.step()
        return self.temperature.item()

    def get_weights_stats(self, weights: Tensor, weights_suffix: str | int) -> Munch:
        out = Munch({
            "min": weights.min().item(),
            "max": weights.max().item(),
            "mean": weights.mean().item(),
            "median": weights.median().item(),
            "quant_75": weights.quantile(0.75).item(),
            "quant_25": weights.quantile(0.25).item(),
        })
        out = Munch({
            f"weights{weights_suffix}/{name}": value for name, value in out.items()
        })
        return out

    def get_weights_info(self, weights: Tensor) -> Munch:
        out = self.get_weights_stats(weights, weights_suffix="")
        for weight_ind in (0, -1):
            out |= self.get_weights_stats(weights[weight_ind], weights_suffix=f"_{weight_ind}")
        return out

    def forward(self):
        temperature = self.get_temperature()
        logits_temperature = self.logits_log_temperature.exp()
        # y_soft = self.get_prompt_logits()
        # y_soft = self.get_prompt_logits() / logits_temperature
        # y_soft = F.gumbel_softmax(self.get_prompt_logits() / logits_temperature, tau=temperature, dim=-1)
        y_soft = F.softmax(self.get_prompt_logits() / logits_temperature, dim=-1)
        # y_soft = F.relu(self.get_prompt_logits() / logits_temperature)
        # y_soft = y_soft / y_soft.sum(dim=-1, keepdim=True)
        y_inds = y_soft.argmax(dim=-1)

        prompts_soft = y_soft @ self.clip_embs
        prompts_hard = self.clip_embs[y_inds, :]
        prompts_hard = straight_through(prompts_hard, prompts_soft)

        out = Munch(
            clip_embs=prompts_soft, gpt_embs=prompts_hard, ids=y_inds.cpu().tolist(),
            temperature=temperature, logits_temperature=logits_temperature.item(),
            **self.get_weights_info(y_soft)
        )
        return out


class Gumbelv0a1(GumbelBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.prompt_logits = nn.Parameter(torch.ones(self.prompt_len, self.clip_embs.shape[0]), requires_grad=True)

    def get_prompt_logits(self):
        return self.prompt_logits

    def step(self) -> Munch:
        return get_prompt_grads_info(self.prompt_logits)


class Gumbelv1a1(GumbelBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        emb_dim = self.clip_embs.shape[1]
        # self.dim_sqrt = emb_dim ** 0.5
        # self.layer_norm = nn.LayerNorm(emb_dim)
        self.prompt_embs = nn.Parameter(torch.randn(self.prompt_len, emb_dim), requires_grad=True)
        nn.init.normal_(self.prompt_embs, std=0.02)

    def get_prompt_logits(self):
        # prompt_embs = self.layer_norm(self.prompt_embs)
        # prompt_embs = self.prompt_embs / self.prompt_embs.norm(dim=1, keepdim=True)
        # prompt_logits = prompt_embs @ self.clip_embs.t() / self.dim_sqrt
        prompt_logits = self.prompt_embs @ self.clip_embs.t()
        return prompt_logits

    def step(self) -> Munch:
        return get_prompt_grads_info(self.prompt_embs)


class Gumbelv3a1(GumbelBase):
    def __init__(self, clip_embs: nn.Embedding, gpt: GPTEmbed, tokenizer, gpt_cfg: DictConfig, **kwargs) -> None:
        super().__init__(clip_embs=clip_embs, **kwargs)
        self.bos_token_emb = clip_embs.weight.data[tokenizer.bos_token_id]  # do not train
        self.gpt = load_obj(gpt_cfg.path)(**gpt_cfg.kwargs, gpt=gpt)

    def select_allowed_tokens(self, logits: Tensor) -> Tensor:
        if self.allowed_tokens is not None:
            logits = logits[:, self.allowed_tokens]
        return logits

    def get_prompt_logits(self):
        past_key_values = None
        prompt_list_logits = []
        input_embs = self.bos_token_emb.unsqueeze(0).unsqueeze(0)  # [1, 1, emb_dim]
        for _ in range(self.prompt_len):
            gpt_out = self.gpt(
                inputs_embeds=input_embs, past_key_values=past_key_values,
                use_cache=True, output_hidden_states=True
            )
            past_key_values = gpt_out.past_key_values
            logits = self.select_allowed_tokens(gpt_out.logits)
            probs = F.softmax(logits, dim=1)  # [1, vocab_dim]
            pred_emb = probs @ self.clip_embs  # [1, emb_dim]
            input_embs = pred_emb.unsqueeze(0)  # [1, 1, emb_dim]
            prompt_list_logits.append(probs.squeeze(0))
        prompt_logits = torch.stack(prompt_list_logits, dim=0)
        return prompt_logits

    @property
    def device(self):
        param = next(iter(self.parameters()))
        return param.device
