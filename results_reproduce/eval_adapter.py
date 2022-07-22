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
from results_reproduce import save_features


def load_checkpoint_state(clip_adapter: train_adapter.ClipAdapter, checkpoint_path: str, device: str) -> train_adapter.ClipAdapter:
    # Since we do not save clip_model, we delete it from the model while loading
    clip_model = clip_adapter.clip_model
    del clip_adapter.clip_model
    clip_adapter = clip_adapter.to(device)

    model_checkpoint = torch.load(checkpoint_path)
    clip_adapter.load_state_dict(model_checkpoint)

    clip_adapter.clip_model = clip_model

    return clip_adapter


def eval_adapter(checkpoint_path: str, visual_encoder_name: str, adapter_fabric: train_adapter.ClipAdapterFabric,
                 dataset_cfg: DictConfig, classes: tp.Optional[tp.List[str]], templates: tp.List[str],
                 image_features_path: str, batch_size: int, num_workers: int, device: str, random_state: int) -> None:
    zero_shot.set_random_state(random_state)

    clip_model, preprocess = clip.load(visual_encoder_name, device)
    dataset = hydra.utils.instantiate(dataset_cfg, transform=preprocess)
    indexed_dataset = save_features.IndexedDataset(dataset)
    loader = DataLoader(indexed_dataset, batch_size=batch_size, num_workers=num_workers)

    clip_adapter = adapter_fabric.create_adapter(clip_model)
    clip_adapter = load_checkpoint_state(clip_adapter, checkpoint_path, device)
    clip_classes = classes or dataset.classes
    text_features = zero_shot.zeroshot_classifier(clip_adapter.clip_model, clip_classes, templates)
    image_features = torch.load(image_features_path)
    clip_adapter_trainer = train_adapter.ClipAdapterTrainer(clip_adapter, image_features, text_features)

    top1, top5 = train_adapter.eval_model(loader, clip_adapter_trainer)
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
        cfg.eval.checkpoint_path, visual_encoder_name, adapter_fabric, cfg.dataset,
        cfg.prompting.classes, cfg.prompting.templates, cfg.eval.image_features_path, cfg.data.batch_size,
        cfg.data.num_workers, cfg.meta.device, cfg.meta.random_state
    )
    logging.info('Finish!')


if __name__ == '__main__':
    run()
