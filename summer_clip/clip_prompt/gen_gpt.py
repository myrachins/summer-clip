import typing as tp
from collections import defaultdict

import torch
import hydra
from torch import nn
from tqdm import tqdm
from datasets.load import load_dataset
from torch.utils.data import DataLoader
from datasets.arrow_dataset import Dataset
from omegaconf import OmegaConf, DictConfig
from transformers import (
    DataCollatorForLanguageModeling, CLIPTokenizer, AutoModelForCausalLM,
    AutoTokenizer, PreTrainedTokenizerBase
)

from summer_clip.utils.hydra_utils import load_obj
from summer_clip.utils.trainer import set_random_state
from summer_clip.clip_prompt.gpt import load_pretrained, ClipGPT
from summer_clip.clip_prompt.tokenize_dataset import tokenize_dataset, tokenize_texts


def create_val_loader(cfg: DictConfig, tokenizer: CLIPTokenizer) -> DataLoader | None:
    if (dt_cfg := cfg.dataset) is None:
        return None
    val_dataset: Dataset = load_dataset(**dt_cfg.dataset)  # type: ignore
    val_filter = hydra.utils.instantiate(dt_cfg.filter)
    val_dataset = val_dataset.filter(val_filter.is_valid, load_from_cache_file=False)
    val_dataset = tokenize_dataset(
        val_dataset, tokenizer, dt_cfg.max_length, dt_cfg.text_column
    )
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    loader = DataLoader(val_dataset, collate_fn=collator, **cfg.data_loader)  # type: ignore
    return loader


def load_pretrained_model(model_cfg_path: str, state_dict_path: str, map_location: tp.Any) -> ClipGPT:
    model_cfg = OmegaConf.load(model_cfg_path)
    assert isinstance(model_cfg, DictConfig)
    state_dict = torch.load(state_dict_path, map_location=map_location)
    return load_pretrained(model_cfg, state_dict)


def load_gpt(model_cfg_path: str) -> AutoModelForCausalLM:
    model_cfg = OmegaConf.load(model_cfg_path)
    model_cls = load_obj(model_cfg.class_path)
    return model_cls.create_gpt(model_cfg)


def load_gpt_tokenizer(model_cfg_path: str) -> PreTrainedTokenizerBase:
    model_cfg = OmegaConf.load(model_cfg_path)
    return AutoTokenizer.from_pretrained(model_cfg.gpt_model_id)


@torch.no_grad()
def evaluate(model: nn.Module, val_loader: DataLoader) -> tuple[float, float]:
    model.eval()
    losses = []
    for batch in tqdm(val_loader):
        outputs = model(**batch)
        losses.append(outputs.loss)
    loss = torch.mean(torch.stack(losses))
    perp = loss.exp()
    return loss.item(), perp.item()


def generate_texts(model: tp.Any, prompts: list[str], tokenizer: CLIPTokenizer, cfg: DictConfig) -> list[list[str]]:
    model.eval()
    input_texts = tokenize_texts(prompts, tokenizer, cfg.generate.max_length)
    gen_texts: list[list[str]] = []
    for input_ids in input_texts['input_ids']:
        input_ids = torch.tensor(input_ids).view(1, -1).to(cfg.meta.device)
        out_ids = model.generate(input_ids, pad_token_id=tokenizer.pad_token_id, **cfg.generate.generate_kwargs)
        out_texts = [tokenizer.decode(text_ids, skip_special_tokens=False) for text_ids in out_ids]
        gen_texts.append(out_texts)
    return gen_texts


def evaluate_lm(model: tp.Any, tokenizer: CLIPTokenizer,
                cfg: DictConfig, res_cfg: defaultdict, model_name: str) -> None:
    val_loader = create_val_loader(cfg, tokenizer)
    model = model.to(cfg.meta.device)
    if val_loader is not None:
        res_cfg['eval'][model_name]['loss'], res_cfg['eval'][model_name]['ppl'] \
            = evaluate(model, val_loader)  # type: ignore
    prompts = cfg.prompts
    if prompts is not None:
        gen_texts = generate_texts(model, prompts, tokenizer, cfg)
        res_prompts = res_cfg.get('prompts', [dict(prompt=prompt) for prompt in prompts])
        for res_prompt, prompt_texts in zip(res_prompts, gen_texts):
            res_prompt[model_name] = prompt_texts  # type: ignore
        res_cfg['prompts'] = res_prompts


def create_inf_defaultdict() -> defaultdict[tp.Any, tp.Any]:
    return defaultdict(create_inf_defaultdict)


def convert_inf_defaultdict(dct: defaultdict[tp.Any, tp.Any]) -> dict[tp.Any, tp.Any]:
    return {
        key: (convert_inf_defaultdict(val) if isinstance(val, defaultdict) else val)
        for key, val in dct.items()
    }


def run_generator(cfg: DictConfig) -> None:
    res_cfg = create_inf_defaultdict()
    clip_tokenizer = CLIPTokenizer.from_pretrained(cfg.tokenizer.tokenizer_id)
    # Do not create separate var for the model. Otherwise GC can not free it before gpt
    evaluate_lm(
        load_pretrained_model(
            cfg.model.meta_cfg_path, cfg.model.state_dict_path, map_location=cfg.meta.device
        ).gpt,
        clip_tokenizer, cfg, res_cfg, model_name='clip_gpt'
    )
    if cfg.eval.eval_gpt:
        gpt_tokenizer = load_gpt_tokenizer(cfg.model.meta_cfg_path)
        gpt_tokenizer.pad_token = gpt_tokenizer.eos_token  # set pad token
        evaluate_lm(
            load_gpt(cfg.model.meta_cfg_path), gpt_tokenizer, cfg, res_cfg, model_name='gpt'  # type: ignore
        )
    res_dict_cfg = convert_inf_defaultdict(res_cfg)
    OmegaConf.save(OmegaConf.create(res_dict_cfg), cfg.data.res_path)


@hydra.main(config_path='../conf', config_name='gen_gpt', version_base='1.1')
def run(cfg: DictConfig) -> None:
    set_random_state(cfg.meta.random_state)
    run_generator(cfg)


if __name__ == '__main__':
    run()
