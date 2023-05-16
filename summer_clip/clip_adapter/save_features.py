import logging
import typing as tp

import torch
import clip
import hydra
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from omegaconf import DictConfig, OmegaConf

from summer_clip.clip_model import eval_clip


class IndexedDataset(Dataset):
    def __init__(self, source_dataset) -> None:
        super().__init__()
        self.source_dataset = source_dataset

    def __len__(self):
        return len(self.source_dataset)

    def __getitem__(self, index):
        image, label = self.source_dataset[index]
        return image, label, index


@torch.no_grad()
def calculate_image_features(model, loader, device):
    images, indexes = [], []
    for image, _, index in tqdm(loader):
        image = image.to(device)
        image_vectors = model.encode_image(image).cpu()
        images.extend(image_vectors)
        indexes.extend(index.tolist())
    images = torch.stack(images, dim=1)
    return images, indexes


def save_image_outs(cfg, clip_model, image_features, dataset, device):
    print('Computing outs...')
    clip_classes = cfg.prompting.classes or dataset.classes
    text_features = eval_clip.zeroshot_classifier(clip_model, clip_classes, cfg.prompting.templates).to(device)
    image_features = image_features.to(device)
    image_features = image_features / image_features.norm(dim=0, keepdim=True)
    image_outs = image_features.t() @ text_features
    torch.save(image_outs, 'train_image_outs.pt')


def save_features(cfg: DictConfig, dataset_cfg: DictConfig, output_path: str, save_outs: bool) -> None:
    eval_clip.set_random_state(cfg.meta.random_state)

    device = cfg.meta.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    clip_model, preprocess = clip.load(cfg.clip.model_name, device, jit=False)
    dataset = hydra.utils.instantiate(dataset_cfg, transform=preprocess)
    indexed_dataset = IndexedDataset(dataset)
    indexed_loader = DataLoader(indexed_dataset, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers)

    images, indexes = calculate_image_features(clip_model, indexed_loader, device)
    assert indexes == list(range(len(indexes))), "Indexes should have consequent order"
    torch.save(images, output_path)

    if save_outs:
        save_image_outs(cfg, clip_model, images, dataset, device)


@hydra.main(config_path='../conf', config_name='save_features', version_base='1.1')
def run(cfg: DictConfig) -> None:
    logging.info('Start!')
    print(OmegaConf.to_yaml(cfg))
    if cfg.get('train_dataset') is not None:
        save_features(cfg, dataset_cfg=cfg.train_dataset, output_path='train_image_features.pt', save_outs=cfg.save_train_outs)
    if cfg.get('test_dataset') is not None:
        save_features(cfg, dataset_cfg=cfg.test_dataset, output_path='test_image_features.pt', save_outs=False)
    logging.info('Finish!')


if __name__ == '__main__':
    run()
