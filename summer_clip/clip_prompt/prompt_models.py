import typing as tp

import torch
from torch import nn
from torch import Tensor


class CoOp(nn.Module):
    def __init__(self, clip_embs: nn.Embedding, prompt_len: int, **kwargs: tp.Any) -> None:
        super().__init__()
        self.prompt_len = prompt_len
        self.prompt_embs = nn.Parameter(torch.randn(prompt_len, clip_embs.weight.shape[1]), requires_grad=True)
        nn.init.normal_(self.prompt_embs, std=0.02)

    def get_clip_prompt_embs(self) -> Tensor:
        return self.prompt_embs

    def get_gpt_prompt_embs(self) -> Tensor:
        return self.prompt_embs

    def get_prompt_ids(self) -> list[int]:
        return [0] * self.prompt_len
