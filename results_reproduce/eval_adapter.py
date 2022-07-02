import logging
import typing as tp
from pathlib import Path

import clip
import hydra
import torch
from torch.utils.data.dataloader import DataLoader
from omegaconf import DictConfig, ListConfig, OmegaConf

from results_reproduce import zero_shot
from results_reproduce import train_adapter


def load_checkpoint_state(clip_adapter: train_adapter.ClipAdapter, checkpoint_path: str, device: str) -> train_adapter.ClipAdapter:
    # Since we do not save clip_model, we delete it from the model while loading
    clip_model = clip_adapter.clip_model
    del clip_adapter.clip_model
    clip_adapter = clip_adapter.to(device)

    model_checkpoint = torch.load(checkpoint_path)
    clip_adapter.load_state_dict(model_checkpoint)

    clip_adapter.clip_model = clip_model
    train_adapter.convert_model_to_fp32(clip_adapter)

    return clip_adapter


def eval_adapter(checkpoint_path: str, visual_encoder_name: str, adapter_fabric: train_adapter.ClipAdapterFabric,
                 dataset_name: str, classes: tp.Optional[tp.List[str]], templates: tp.List[str],
                 batch_size: int = 32, num_workers: int = 2, device: str = 'cuda', random_state: int = 42) -> None:
    zero_shot.set_random_state(random_state)

    clip_model, preprocess = clip.load(visual_encoder_name, device)
    dataset = zero_shot.get_dataset(dataset_name, preprocess)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    classes, templates = zero_shot.load_promts(dataset_name)

    clip_adapter = adapter_fabric.create_adapter(clip_model)
    clip_adapter = load_checkpoint_state(clip_adapter, checkpoint_path, device)

    zeroshot_weights = zero_shot.zeroshot_classifier(clip_adapter, classes, templates)
    top1, top5 = zero_shot.compute_accuracy(clip_adapter, zeroshot_weights, loader)
    logging.info(f'acc@1: {top1}')
    logging.info(f'acc@5: {top5}')


def load_train_config(eval_cfg: DictConfig) -> tp.Union[DictConfig, ListConfig]:
    if eval_cfg.eval.train_config_path:
        return OmegaConf.load(eval_cfg.eval.train_config_path)

    train_config_path = Path(eval_cfg.eval.checkpoint_path).parent.parent.parent / '.hydra' / 'config.yaml'
    return OmegaConf.load(train_config_path)


@hydra.main(config_path='conf', config_name='eval_adapter', version_base='1.1')
def run(cfg: DictConfig) -> None:
    logging.info('Start!')
    print(OmegaConf.to_yaml(cfg))

    train_cfg = load_train_config(cfg)
    visual_encoder_name = train_cfg.clip.model_name
    adapter_fabric = hydra.utils.instantiate(train_cfg.adapter)

    eval_adapter(
        cfg.eval.checkpoint_path, visual_encoder_name, adapter_fabric, cfg.dataset.dataset_name,
        cfg.prompting.classes, cfg.prompting.templates, cfg.dataset.batch_size, cfg.dataset.num_workers,
        cfg.meta.device, cfg.meta.random_state
    )
    logging.info('Finish!')


if __name__ == '__main__':
    run()
