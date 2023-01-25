import typing as tp
import hydra
from omegaconf import DictConfig
from datasets.load import load_dataset
from transformers import CLIPTokenizer
from datasets.arrow_dataset import Dataset
from summer_clip.utils.trainer import set_random_state


def tokenize_dataset(dataset: Dataset, tokenizer: CLIPTokenizer, max_length: int, text_column: str, num_proc: tp.Optional[int]) -> Dataset:
    def tokenization(example):
        texts = ["<|startoftext|>" + text for text in example[text_column]]
        return tokenizer(texts, add_special_tokens=False, truncation=True, max_length=max_length)

    encodings = dataset.map(tokenization, batched=True, num_proc=num_proc, remove_columns=dataset.column_names)
    return encodings


def run_tokenizer(cfg: DictConfig) -> None:
    print('Loading tokenizer & dataset...')
    tokenizer = CLIPTokenizer.from_pretrained(cfg.clip.tokenizer_id)
    dataset: Dataset = load_dataset(**cfg.dataset.dataset)  # type: ignore
    print('Tokenization...')
    dataset = tokenize_dataset(
        dataset, tokenizer, cfg.dataset.max_length, cfg.dataset.text_column, cfg.dataset.num_proc
    )
    print('Saving tokenized dataset...')
    dataset.save_to_disk(**cfg.save_kwargs)
    print('Dataset was saved!')


@hydra.main(config_path='../conf', config_name='tokenize_dataset', version_base='1.1')
def run(cfg: DictConfig) -> None:
    set_random_state(cfg.meta.random_state)
    run_tokenizer(cfg)


if __name__ == '__main__':
    run()
