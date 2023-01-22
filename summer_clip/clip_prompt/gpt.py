import itertools
import typing as tp
from dataclasses import dataclass, asdict

import torch
from torch import nn
from transformers import GPT2LMHeadModel, GPT2Config


class Adapter(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hid_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim, bias=False),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class AdapterEmb(nn.Module):
    def __init__(self, emb: nn.Embedding, adapter: Adapter) -> None:
        super().__init__()
        self.emb = emb
        self.adapter = adapter

    def forward(self, input_ids):
        x = self.emb(input_ids)
        x = self.adapter(x)
        return x


class AdapterLMHead(nn.Module):
    def __init__(self, emb: nn.Embedding, adapter: Adapter) -> None:
        super().__init__()
        self.lm_head = nn.Linear(emb.embedding_dim, emb.num_embeddings, bias=False)
        self.lm_head.weight = emb.weight
        self.adapter = adapter

    def forward(self, x):
        x = self.lm_head(x)
        x = self.adapter(x)
        return x


@dataclass
class ClipGPTConfig:
    adap_emb_hid_dim: int
    adap_head_hid_dim: tp.Optional[int]  # if null: head & emb are tied


class ClipGPT(nn.Module):
    def __init__(self, cfg: ClipGPTConfig, clip_emb: nn.Embedding, gpt: GPT2LMHeadModel) -> None:
        super().__init__()
        self.cfg = cfg
        self.clip_emb = clip_emb
        self.gpt = gpt

        self._add_adapters()
        self._set_grads()

    @classmethod
    def from_scratch(cls) -> 'ClipGPT':
        cfg = ClipGPTConfig(adap_emb_hid_dim=1, adap_head_hid_dim=None)
        clip_emb = nn.Embedding(1, 1)
        gpt = GPT2LMHeadModel(GPT2Config())
        return cls(cfg, clip_emb, gpt)

    @classmethod
    def from_pretrained(cls, training_state_dict: dict[str, tp.Any], clip_emb: nn.Embedding, gpt: GPT2LMHeadModel) -> 'ClipGPT':
        cfg = ClipGPTConfig(**training_state_dict['cfg'])
        model = cls(cfg, clip_emb, gpt)
        model.load_state_dict(training_state_dict, strict=False)
        return model

    def _add_adapters(self):
        clip_emb_dim = self.clip_emb.embedding_dim
        gpt_emb_dim: int = self.gpt.get_input_embeddings().embedding_dim  # type: ignore

        emb_adap = Adapter(
            in_dim=clip_emb_dim,
            hid_dim=self.cfg.adap_emb_hid_dim,
            out_dim=gpt_emb_dim
        )
        self.gpt.set_input_embeddings(AdapterEmb(self.clip_emb, emb_adap))
        head_adap = Adapter(
            in_dim=clip_emb_dim,
            hid_dim=self.cfg.adap_head_hid_dim,
            out_dim=gpt_emb_dim
        ) if (self.cfg.adap_head_hid_dim is not None) else emb_adap
        self.gpt.set_output_embeddings(AdapterLMHead(self.clip_emb, head_adap))

    def _set_grads(self):
        for param in self.parameters():
            param.requires_grad_(False)
        for param in self.training_parameters():
            param.requires_grad_(True)

    def training_parameters(self):
        return (param for _, param in self.named_training_parameters())

    def named_training_parameters(self):
        return itertools.chain(
            self.gpt.get_input_embeddings().adapter.named_parameters(),  # type: ignore
            self.gpt.get_output_embeddings().adapter.named_parameters(),  # type: ignore
        )

    def training_state_dict(self):
        params_dict = dict(self.named_training_parameters())
        return asdict(self.cfg) | params_dict

    def forward(self, *args, **kwargs):
        x = self.gpt(*args, **kwargs)
        return x
