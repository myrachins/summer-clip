import typing as tp
from abc import ABC, abstractmethod

from torch import nn
from transformers import CLIPTokenizer


class BaseVocabFilter(ABC):
    def __init__(self, clip_embs: nn.Embedding, clip_tokenizer: CLIPTokenizer, **kwargs: tp.Any) -> None:
        self.clip_embs = clip_embs
        self.clip_tokenizer = clip_tokenizer

    def tokenize_tokens(self, tokens: list[str]) -> tp.Any:
        ids = self.clip_tokenizer(
            list(tokens), add_special_tokens=False, is_split_into_words=True
        )['input_ids']
        return ids

    @abstractmethod
    def get_allowed_tokens(self):
        """
        Returns ids of the tokens
        """


class NoFilter(BaseVocabFilter):
    def get_allowed_tokens(self):
        return None


class AllowedTokensFilter(BaseVocabFilter):
    def __init__(self, allowed_tokens: list[str], check_tokens_single: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.tokens_ids = self.tokenize_tokens(allowed_tokens)  # type: ignore

        if check_tokens_single and len(self.tokens_ids) != len(allowed_tokens):
            raise ValueError("Lens of the ids and tokens do not match")

    def get_allowed_tokens(self):
        return self.tokens_ids


class NotAllowedTokensFilter(BaseVocabFilter):
    def __init__(self, not_allowed_tokens: list[str], **kwargs):
        super().__init__(**kwargs)
        clip_embs_num = self.clip_embs.weight.shape[0]
        not_allowed_ids = self.tokenize_tokens(not_allowed_tokens)
        self.allowed_ids = list(set(range(clip_embs_num)) - set(not_allowed_ids))

    def get_allowed_tokens(self):
        return self.allowed_ids
