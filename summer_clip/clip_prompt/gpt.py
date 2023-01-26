import itertools
import typing as tp

import clip
import torch.nn.functional as F
from torch import nn
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM

from summer_clip.utils.hydra_utils import load_obj


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
        self.lm_head = nn.Parameter(emb.weight)
        self.adapter = adapter

    def forward(self, x):
        lm_head = self.adapter(self.lm_head)
        x = F.linear(x, lm_head)
        return x


class ClipGPT(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.clip_emb = self._create_clip_emb(cfg)
        self.gpt = self._create_gpt(cfg)

        self._add_adapters()
        self._set_grads()

    @staticmethod
    def _create_clip_emb(cfg):
        clip_model, _ = clip.load(cfg.clip_model_name, device='cpu', jit=False)
        return clip_model.token_embedding

    @staticmethod
    def _create_gpt(cfg):
        return AutoModelForCausalLM.from_pretrained(cfg.gpt_model_id)

    def _add_adapters(self):
        clip_emb_dim = self.clip_emb.embedding_dim
        gpt_emb_dim: int = self.gpt.get_input_embeddings().embedding_dim  # type: ignore

        emb_adap = Adapter(
            in_dim=clip_emb_dim,
            hid_dim=self.cfg.adapters.emb_hid_dim,
            out_dim=gpt_emb_dim
        )
        self.gpt.set_input_embeddings(AdapterEmb(self.clip_emb, emb_adap))
        head_adap = Adapter(
            in_dim=clip_emb_dim,
            hid_dim=self.cfg.adapters.head_hid_dim,
            out_dim=gpt_emb_dim
        ) if (self.cfg.adapters.head_hid_dim is not None) else emb_adap
        self.gpt.set_output_embeddings(AdapterLMHead(self.clip_emb, head_adap))

    def _set_grads(self):
        for param in self.parameters():
            param.requires_grad_(False)
        for param in self.training_parameters():
            param.requires_grad_(True)

    def training_parameters(self):
        return itertools.chain(
            self.gpt.get_input_embeddings().adapter.parameters(),  # type: ignore
            self.gpt.get_output_embeddings().adapter.parameters(),  # type: ignore
        )

    def named_training_parameters(self):
        return ((name, param) for name, param in self.named_parameters() if param.requires_grad)

    def training_state_dict(self):
        return dict(self.named_training_parameters())

    def forward(self, *args, **kwargs):
        x = self.gpt(*args, **kwargs)
        return x


def load_model(model_cfg: DictConfig) -> tp.Any:
    model_cls = load_obj(model_cfg.class_path)
    model = model_cls(model_cfg)
    return model


def load_pretrained(model_cfg: DictConfig, training_state_dict: dict[str, tp.Any]) -> tp.Any:
    model = load_model(model_cfg)
    model.load_state_dict(training_state_dict, strict=False)
    return model


class ClipGPTFull(ClipGPT):
    def _set_grads(self):
        for param in self.parameters():
            param.requires_grad_(True)
        for param in self._non_training_parameters():
            param.requires_grad_(False)

    def _non_training_parameters(self):
        return itertools.chain(
            self.clip_emb.parameters(),
            self.gpt.get_input_embeddings().emb.parameters(),  # type: ignore
            self.gpt.get_output_embeddings().lm_head.parameters(),  # type: ignore
        )

    def training_parameters(self):
        return (param for _, param in self.named_training_parameters())
