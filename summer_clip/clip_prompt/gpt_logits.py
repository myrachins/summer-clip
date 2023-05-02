import typing as tp

from torch import nn
from torch.nn import functional as F
from peft import get_peft_model, LoraConfig, TaskType

from summer_clip.clip_prompt.prompt_learner import GPTEmbed


class EmbsAdapter(nn.Module):
    def __init__(self, embs_dim: int, hidden_dim: int):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Linear(embs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embs_dim),
        )
        self.init_blocks()

    def init_blocks(self):
        # Is based on RL-Prompt
        def init_weights(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.0001)
                module.bias.data.fill_(-0.0001)
        self.blocks.apply(init_weights)

    def forward(self, x):
        y = self.blocks(x)
        y = y + x
        return y


class NotTrainable:
    def __init__(self, param):
        self.param = param

    def __getattr__(self, name: str):
        return getattr(self.param, name)


class AdapterGPT(nn.Module):
    def __init__(self, gpt: GPTEmbed, hidden_dim: int):
        super().__init__()
        self.gpt: tp.Any = NotTrainable(gpt)
        gpt_emb_dim: int = gpt.gpt.config.n_embd  # type: ignore
        self.adapter = EmbsAdapter(gpt_emb_dim, hidden_dim)

    def forward(self, **kwargs):
        gpt_out = self.gpt.__call__(**kwargs)
        hidden_state = gpt_out.hidden_states[-1][:, -1, :]
        hidden_state = self.adapter(hidden_state)
        logits = self.gpt.gpt.lm_head(hidden_state)
        gpt_out.logits = logits
        return gpt_out


class LoRAGPT(nn.Module):
    def __init__(self, gpt: GPTEmbed, **kwargs):
        super().__init__()
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, **kwargs)
        gpt.gpt = get_peft_model(gpt.gpt, peft_config)
        self.gpt = gpt

    def forward(self, **kwargs):
        gpt_out = self.gpt(**kwargs)
        gpt_out.logits = gpt_out.logits[:, -1, :]
        return gpt_out
