import typing as tp

import torch
from torch import nn
from torch import Tensor
from munch import Munch
from torch.nn import functional as F


def find_nearest(prompt_embs: Tensor, clip_embs: Tensor, p: float) -> Tensor:
    dists = torch.cdist(
        prompt_embs.unsqueeze(0), clip_embs.unsqueeze(0), p=p
    ).squeeze(0)
    prompt_ids = dists.argmin(dim=1)
    return prompt_ids


def straight_through(out_val: Tensor, out_grad: Tensor) -> Tensor:
    out = (out_val - out_grad).detach() + out_grad
    return out


class CoOp(nn.Module):
    def __init__(self, clip_embs: nn.Embedding, prompt_len: int, dist_p: float, **kwargs: tp.Any) -> None:
        super().__init__()
        self.dist_p = dist_p
        self.prompt_len = prompt_len
        self.clip_embs = clip_embs.weight.data  # we are not training this one
        self.prompt_embs = nn.Parameter(torch.randn(prompt_len, clip_embs.weight.shape[1]), requires_grad=True)
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


class VQVAE1(nn.Module):
    def __init__(self, clip_embs: nn.Embedding, prompt_len: int, dist_p: float, **kwargs: tp.Any) -> None:
        super().__init__()
        self.dist_p = dist_p
        self.clip_embs = clip_embs.weight.data  # we are not training this one
        self.prompt_embs = nn.Parameter(torch.randn(prompt_len, clip_embs.weight.shape[1]), requires_grad=True)
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


class Gumbelv0a1(nn.Module):
    def __init__(self, clip_embs: nn.Embedding, prompt_len: int, **kwargs: tp.Any) -> None:
        self.clip_embs = clip_embs.weight.data  # we are not training this one
        self.prompt_logits = nn.Parameter(torch.ones(prompt_len, clip_embs.weight.shape[0]), requires_grad=True)

    def forward(self):
        # TODO: Add temperature
        y_soft = F.gumbel_softmax(self.prompt_logits, dim=-1)
        y_inds = y_soft.argmax(dim=-1)

        prompts_soft = y_soft @ self.clip_embs
        prompts_hard = self.clip_embs[y_inds, :]
        prompts_hard = straight_through(prompts_hard, prompts_soft)

        out = Munch(
            clip_embs=prompts_soft, gpt_embs=prompts_hard, ids=y_inds.cpu().tolist()
        )
        return out
