import random
import typing as tp

import torch
from torch import nn
from torch import Tensor
from transformers import DataCollatorForLanguageModeling

from summer_clip.clip_prompt.gpt import ClipGPT


class GPTEmbed(nn.Module):
    def __init__(self, gpt: ClipGPT) -> None:
        super().__init__()
        self.gpt = gpt

    def forward(self, inputs_embeds, **kwargs):
        inputs_embeds = self.gpt.transformer.wte.adapter(inputs_embeds)  # type: ignore
        return self.gpt(inputs_embeds=inputs_embeds, **kwargs)


class ClipTextEncoder(nn.Module):
    def __init__(self, clip_model: tp.Any) -> None:
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

    def forward(self, inputs_embeds, input_lens, **kwargs):
        x = inputs_embeds + self.positional_embedding.unsqueeze(0)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        # x.shape = [batch_size, n_ctx, transformer.width]
        last_token_ids = [input_len - 1 for input_len in input_lens]
        x = x[torch.arange(x.shape[0]), last_token_ids] @ self.text_projection
        return x


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
    def __init__(self, tokenizer, embs, clip_seq_len) -> None:
        self.tokenizer = tokenizer
        self.embs = embs

        self.bos_id = self.tokenizer.bos_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.lm_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        self.clip_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=clip_seq_len)

    def _create_batch(self, input_ids, prompt_embs, collator):
        batch = [dict(input_ids=i_ids, attention_mask=[1] * len(i_ids)) for i_ids in input_ids]
        batch = collator(batch)
        batch = {key: val.to(prompt_embs.device) for key, val in batch.items()}
        input_embs = self.embs(batch['input_ids'])
        input_embs[:, 1:prompt_embs.shape[0] + 1, :] = prompt_embs.unsqueeze(0)
        batch['inputs_embeds'] = input_embs
        return batch

    def get_gpt_input(self, prompt_embs, prompt_ids, input_ids):
        prompt_ids = list(prompt_ids)
        input_ids = [
            [self.bos_id] + prompt_ids + list(i_ids)
            for i_ids in input_ids
        ]
        lm_batch = self._create_batch(input_ids, prompt_embs, self.lm_collator)
        lm_batch.pop('input_ids')
        return lm_batch

    def get_clip_input(self, prompt_embs, prompt_ids, input_ids):
        prompt_ids = list(prompt_ids)
        input_ids = [
            [self.bos_id] + prompt_ids + list(i_ids) + [self.eos_id]
            for i_ids in input_ids
        ]
        clip_batch = self._create_batch(input_ids, prompt_embs, self.clip_collator)
        clip_batch['input_lens'] = [len(i_ids) for i_ids in input_ids]
        return clip_batch


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


class FullLMLoss:
    def transform(self, lm_in, lm_out) -> Tensor:
        return lm_out.loss


class SuffixLMLoss:
    def __init__(self, prompt_len: int, has_bos: bool = True):
        self.prefix_len = prompt_len
        if has_bos:
            self.prefix_len += 1
        self.loss = nn.CrossEntropyLoss()

    def transform(self, lm_in, lm_out) -> Tensor:
        # Since labels == input_ids, we make the shift manually
        # https://github.com/huggingface/transformers/blob/68287689f2f0d8b7063c400230b3766987abf18d/src/transformers/models/gpt2/modeling_gpt2.py#L1100
        logits = lm_out['logits'][..., self.prefix_len:-1, :].contiguous()
        labels = lm_in['labels'][..., self.prefix_len + 1:].contiguous()
        loss = self.loss(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss


class NoLMLoss:
    def transform(self, lm_in, lm_out) -> Tensor:
        loss = torch.zeros_like(lm_out.loss)
        return loss
