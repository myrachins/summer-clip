import re
import string
import typing as tp
from abc import ABC, abstractmethod

from torch import nn
from omegaconf import OmegaConf
from transformers import CLIPTokenizer


class BaseVocabFilter(ABC):
    def __init__(self, clip_embs: nn.Embedding, clip_tokenizer: CLIPTokenizer, **kwargs: tp.Any) -> None:
        self.clip_embs = clip_embs
        self.clip_tokenizer = clip_tokenizer

    def tokenize_tokens(self, tokens: list[str]) -> list[int]:
        ids = [self.clip_tokenizer.encoder[token] for token in tokens]
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
    def __init__(self, allowed_tokens: list[str], check_unique: bool = True, **kwargs):
        super().__init__(**kwargs)
        if check_unique and len(set(allowed_tokens)) != len(allowed_tokens):
            raise ValueError("Some of the tokens are duplicated")
        self.tokens_ids = self.tokenize_tokens(allowed_tokens)  # type: ignore

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


class FilterNonBasicStrong(BaseVocabFilter):
    def __init__(self, keep_english: bool, keep_numbers: bool, keep_punctuation: bool, **kwargs):
        super().__init__(**kwargs)
        patterns = []
        if keep_english:
            patterns.append(r"[a-zA-Z]+")
        if keep_numbers:
            patterns.append(r"[0-9]+")
        if keep_punctuation:
            patterns.append(f"[{re.escape(string.punctuation)}]+")
        pattern = re.compile("^(" + "|".join(patterns) + ")$")

        allowed_tokens = [
            token for token in self.clip_tokenizer.encoder
            if pattern.match(self.clean_suffix(token))
        ]
        self.filter = AllowedTokensFilter(allowed_tokens, **kwargs)

    def clean_suffix(self, token):
        suffix = "</w>"
        if token.endswith(suffix):
            token = token[:-len(suffix)]
        return token

    def get_allowed_tokens(self):
        return self.filter.get_allowed_tokens()


class PromptsUnionFilter(BaseVocabFilter):
    def __init__(self, prompts_paths: tuple[str, ...] = (), classes_paths: tuple[str, ...] = (), **kwargs):
        super().__init__(**kwargs)
        prompts, classes = [], []
        for prompts_path in prompts_paths:
            path_prompts = OmegaConf.load(prompts_path)['templates']  # type: ignore
            prompts += [prompt.format('') for prompt in path_prompts]
        for classes_path in classes_paths:
            classes += list(OmegaConf.load(classes_path)['classes'])  # type: ignore

        union_texts = prompts + classes
        union_ids: tp.Any = self.clip_tokenizer(union_texts, add_special_tokens=False)['input_ids']
        self.union_ids = list({token_ids for sent_ids in union_ids for token_ids in sent_ids})

    def get_allowed_tokens(self):
        return self.union_ids
