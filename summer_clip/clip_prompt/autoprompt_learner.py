import random
from copy import copy
import typing as tp

import torch
from torch import nn
from torch import optim
from torch import Tensor
from omegaconf import DictConfig
from torch.utils.data.dataloader import DataLoader


class DummyOptimizer(optim.Optimizer):
    def step(self):
        pass


class DummyScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, **kwargs):
        super().__init__(optimizer)

    def step(self):
        pass


def hotflip_attack(averaged_grad: Tensor, embedding_matrix: Tensor, num_cands: int) -> list[int]:
    """Returns the top candidate replacements."""
    with torch.no_grad():
        gradient_dot_embedding_matrix = torch.matmul(
            embedding_matrix,
            averaged_grad
        )
        gradient_dot_embedding_matrix *= -1
        _, top_k_ids = gradient_dot_embedding_matrix.topk(num_cands)

    top_k_ids = top_k_ids.cpu().tolist()
    return top_k_ids


class AutoPromptModel(nn.Module):
    def __init__(self, model_cfg: DictConfig, trainer: tp.Any, clip_embs: nn.Embedding, init_ids: list[int]) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.gpt = trainer.gpt
        self.text_batcher = trainer.text_batcher
        self.collator = trainer.collator
        self.clip_embs = clip_embs.weight.data
        self.prompt_ids = init_ids
        self.loader = trainer.loaders['train']
        self.prompt_embs = nn.Parameter(self.clip_embs[init_ids].detach().clone(), requires_grad=True)

    def get_prompt_embs(self) -> torch.Tensor:
        return self.prompt_embs

    def get_prompt_ids(self) -> list[int]:
        return self.prompt_ids

    def _compute_lm_loss(self, labels, prompt_embs, prompt_ids):
        batch_classes = self.text_batcher.get_batch_classes(labels)
        lm_batch = self.collator.get_gpt_input(
            prompt_embs=prompt_embs, prompt_ids=prompt_ids,
            input_ids=batch_classes
        )
        loss = self.gpt(**lm_batch).loss
        return loss

    def step(self):
        token_to_flip = random.randrange(self.prompt_embs.shape[0])
        token_to_flip_grad = self.prompt_embs.grad[token_to_flip]  # type: ignore
        candidates = hotflip_attack(
            token_to_flip_grad, self.clip_embs, num_cands=self.model_cfg.num_cands
        )
        device = self.clip_embs.device
        curr_loss = 0.
        cand_losses = torch.zeros(self.model_cfg.num_cands, device=device)
        train_iter = iter(self.loader)

        for _ in range(self.model_cfg.search_steps):
            labels, _ = next(train_iter)
            with torch.no_grad():
                curr_loss += self._compute_lm_loss(
                    labels, self.get_prompt_embs(), self.get_prompt_ids()
                )
            for cand_ind, cand in enumerate(candidates):
                cand_ids = copy(self.get_prompt_ids())
                cand_embs = self.get_prompt_embs().clone()
                cand_ids[token_to_flip] = cand
                cand_embs[token_to_flip] = self.clip_embs[cand].detach().clone()
                with torch.no_grad():
                    cand_losses[cand_ind] += self._compute_lm_loss(
                        labels, cand_embs, cand_ids
                    )

        best_cand = candidates[cand_losses.argmin()]
        best_cand_loss = cand_losses.min()
        if best_cand_loss < curr_loss:
            self.prompt_ids[token_to_flip] = best_cand
            self.prompt_embs.data[token_to_flip] = self.clip_embs[best_cand].detach().clone()
