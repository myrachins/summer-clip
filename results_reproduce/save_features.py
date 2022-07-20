import logging

import torch
import clip
import hydra
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from omegaconf import DictConfig, OmegaConf

from results_reproduce import zero_shot


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
        image_vectors = model.encode_image(image)
        images.extend(image_vectors)
        indexes.extend(index.tolist())
    images = torch.stack(images, dim=1)
    return images, indexes


def save_features(model_name: str, dataset_name: str, output_path: str, batch_size: int, num_workers: int,
                  device: str, random_state: int) -> None:
    zero_shot.set_random_state(random_state)

    clip_model, preprocess = clip.load(model_name, device, jit=False)
    dataset = zero_shot.get_dataset(dataset_name, preprocess)
    indexed_dataset = IndexedDataset(dataset)
    indexed_loader = DataLoader(indexed_dataset, batch_size=batch_size, num_workers=num_workers)

    images, indexes = calculate_image_features(clip_model, indexed_loader, device)
    assert indexes == list(range(len(indexes))), "Indexes should have consequent order"
    torch.save(images, output_path)


@hydra.main(config_path='conf', config_name='save_features', version_base='1.1')
def run(cfg: DictConfig) -> None:
    logging.info('Start!')
    print(OmegaConf.to_yaml(cfg))
    save_features(
        cfg.clip.model_name, cfg.dataset.dataset_name, cfg.data.output_path, cfg.dataset.batch_size,
        cfg.dataset.num_workers, cfg.meta.device, cfg.meta.random_state
    )
    logging.info('Finish!')


if __name__ == '__main__':
    run()
