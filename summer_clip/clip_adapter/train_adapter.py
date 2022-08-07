import itertools
import logging
import typing as tp
from pathlib import Path
from copy import copy
from abc import ABC, abstractmethod

import torch
import torch.utils
import torch.optim
import clip
import hydra
import wandb
import numpy as np
from torch import nn
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataset import random_split
from torch.utils.data.dataloader import DataLoader

from summer_clip.clip_model import eval_clip


class ClipAdapter(nn.Module):
    def __init__(self, clip_model, vision_adapter, text_adapter):
        super().__init__()
        self.clip_model = clip_model
        self.vision_adapter = vision_adapter
        self.text_adapter = text_adapter

    def encode_image(self, image):
        image = self.clip_model.encode_image(image)
        image = self.vision_adapter(image)
        return image

    def encode_text(self, text):
        text = self.clip_model.encode_text(text)
        text = self.text_adapter(text)
        return text


class ClipAdapterTrainer(nn.Module):
    def __init__(self, clip_adapter: ClipAdapter, image_features: torch.Tensor, text_features: torch.Tensor) -> None:
        super().__init__()
        self.clip_adapter = clip_adapter
        self.image_features = image_features.float()
        self.text_features = text_features.float()

    def forward(self, index, label):
        with torch.no_grad():
            image_features = self.image_features[:, index].t()
            text_features = self.text_features[:, label].t()

        image_features = self.clip_adapter.vision_adapter(image_features)
        text_features = self.clip_adapter.text_adapter(text_features)

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        logit_scale = self.clip_adapter.clip_model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text


class ClipAdapterFabric(ABC):
    @abstractmethod
    def create_adapter(self, clip_model) -> ClipAdapter:
        pass


class LinearClipAdapterFabric(ClipAdapterFabric):
    def __init__(self, output_dim: tp.Optional[int] = None) -> None:
        super().__init__()
        self.output_dim = output_dim

    def create_adapter(self, clip_model) -> ClipAdapter:
        embed_dim = clip_model.text_projection.size(dim=1)
        output_dim = self.output_dim or embed_dim
        vision_adapter = nn.Linear(embed_dim, output_dim)
        text_adapter = nn.Linear(embed_dim, output_dim)
        return ClipAdapter(clip_model, vision_adapter, text_adapter)


class OriginalClipAdapter(nn.Module):
    def __init__(self, input_dim, dim_reduction, res_ratio):
        super().__init__()
        middle_dim = input_dim // dim_reduction
        self.fc = nn.Sequential(
            nn.Linear(input_dim, middle_dim, bias=False),
            nn.ReLU(),
            nn.Linear(middle_dim, input_dim, bias=False),
            nn.ReLU()
        )
        self.res_ratio = res_ratio

    def forward(self, x):
        x = x / x.norm(dim=1, keepdim=True)
        x = self.res_ratio * self.fc(x) + (1 - self.res_ratio) * x
        return x


class OriginalImageClipAdapterFabric(ClipAdapterFabric):
    def __init__(self, dim_reduction, res_ratio) -> None:
        super().__init__()
        self.dim_reduction = dim_reduction
        self.res_ratio = res_ratio

    def create_adapter(self, clip_model) -> ClipAdapter:
        input_dim = clip_model.text_projection.size(dim=1)
        vision_adapter = OriginalClipAdapter(input_dim, self.dim_reduction, self.res_ratio)
        text_adapter = nn.Identity()
        return ClipAdapter(clip_model, vision_adapter, text_adapter)


class NoImageIndexedDataset(Dataset):
    def __init__(self, source_dataset) -> None:
        super().__init__()
        self.source_dataset = source_dataset

    def __len__(self):
        return len(self.source_dataset)

    def __getitem__(self, index):
        _, label = self.source_dataset[index]
        return label, index


def train_epoch(loader: DataLoader, model: ClipAdapterTrainer, loss: nn.CrossEntropyLoss, optimizer: torch.optim.Optimizer,
                device: torch.device):
    model.train()
    model = model.to(device)
    epoch_loss = 0.

    for labels, indexes in tqdm(loader):
        model.zero_grad()
        indexes = indexes.to(device)
        labels = labels.to(device)

        logits_per_image, logits_per_text = model(indexes, labels)
        dummy_labels = torch.arange(len(labels), device=device)
        image_loss = loss(logits_per_image, dummy_labels)
        text_loss = loss(logits_per_text, dummy_labels)
        agg_loss = (image_loss + text_loss) / 2

        wandb.log({
            'loss-train-image': image_loss.item(),
            'loss-train-text': text_loss.item(),
            'loss-train-agg': agg_loss.item(),
        })
        epoch_loss += agg_loss.item()

        agg_loss.backward()
        optimizer.step()

    return model, epoch_loss, optimizer


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


@torch.no_grad()
def compute_accuracy(image_features, text_features, loader):
    image_features = image_features / image_features.norm(dim=0, keepdim=True)
    text_features = text_features / text_features.norm(dim=0, keepdim=True)

    top1, top5, n = 0., 0., 0.
    for target, index in tqdm(loader):
        # predict
        batch_image_features = image_features[:, index]
        logits = 100. * batch_image_features.t() @ text_features

        # measure accuracy
        target = target.to(logits.device)
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        top1 += acc1
        top5 += acc5
        n += target.size(0)

    if n <= 0:
        return np.nan, np.nan

    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100

    return top1, top5


def eval_model(loader: DataLoader, model: ClipAdapterTrainer) -> tp.Tuple[float, float]:
    model.eval()
    image_features = model.clip_adapter.vision_adapter(model.image_features.t()).t()
    text_features = model.clip_adapter.text_adapter(model.text_features.t()).t()
    return compute_accuracy(image_features, text_features, loader)


def save_epoch_model(model: ClipAdapter, optimizer: torch.optim.Optimizer, epoch_num: int, checkpoints_dir: Path):
    epoch_dir = checkpoints_dir / f'epoch_{epoch_num}'
    epoch_dir.mkdir(parents=True, exist_ok=True)

    def save_data(data, data_name):
        with open(epoch_dir / f'{data_name}.ckpt', 'wb') as f:
            torch.save(data, f)

    model_state_dict = copy(model.state_dict())
    clip_model_keys = [param_name for param_name in model_state_dict.keys() if param_name.startswith('clip_model')]
    for clip_model_key in clip_model_keys:
        model_state_dict.pop(clip_model_key)

    save_data(model_state_dict, 'model')
    save_data(optimizer.state_dict(), 'optimizer')


def train_model(train_loader: DataLoader, val_loader: DataLoader, model: ClipAdapterTrainer, loss: nn.CrossEntropyLoss,
                optimizer: torch.optim.Optimizer, device: torch.device, epochs_num: int, checkpoints_dir: Path):
    for epoch_num in range(1, epochs_num + 1):
        print(f'Running epoch {epoch_num}...')
        model, epoch_loss, optimizer = train_epoch(train_loader, model, loss, optimizer, device)
        print('Evaluating model on train...')
        eval_top1, eval_top5 = eval_model(train_loader, model)
        wandb.log({
            'train-epoch-sum-loss': epoch_loss,
            'train-epoch-acc@1': eval_top1,
            'train-epoch-acc@5': eval_top5,
        })
        logging.info(f'{epoch_num=}, train-acc@1: {eval_top1}')
        logging.info(f'{epoch_num=}, train-acc@5: {eval_top5}')
        print('Evaluating model on validation...')
        eval_top1, eval_top5 = eval_model(val_loader, model)
        wandb.log({
            'val-epoch-acc@1': eval_top1,
            'val-epoch-acc@5': eval_top5,
        })
        logging.info(f'{epoch_num=}, val-acc@1: {eval_top1}')
        logging.info(f'{epoch_num=}, val-acc@5: {eval_top5}')
        print(f'Saving checkpoint after {epoch_num} epoch...')
        save_epoch_model(model.clip_adapter, optimizer, epoch_num, checkpoints_dir)


def train_val_split(dataset, validation_size: float):
    val_size = int(len(dataset) * validation_size)
    train_size = len(dataset) - val_size
    return random_split(dataset, [train_size, val_size])


def train_adapter(model_name: str, dataset_cfg: DictConfig, validation_size: float, batch_size: int, num_workers: int,
                  adapter_fabric: ClipAdapterFabric, classes: tp.Optional[tp.List[str]], templates: tp.List[str],
                  image_features_path: str, adam_params: tp.Dict[str, tp.Any], epochs_num: int,
                  checkpoints_dir: str, device: str, random_state: int) -> None:
    eval_clip.set_random_state(random_state)
    torch_device = torch.device(device)
    checkpoints_path = Path(checkpoints_dir)

    clip_model, _ = clip.load(model_name, device, jit=False)
    dataset = hydra.utils.instantiate(dataset_cfg)
    indexed_dataset = NoImageIndexedDataset(dataset)
    train_dataset, val_dataset = train_val_split(indexed_dataset, validation_size)
    logging.info(f'train-size={len(train_dataset)}, val-size={len(val_dataset)}')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

    clip_adapter = adapter_fabric.create_adapter(clip_model)
    clip_classes = classes or dataset.classes
    text_features = eval_clip.zeroshot_classifier(clip_adapter.clip_model, clip_classes, templates)
    image_features = torch.load(image_features_path)
    clip_adapter_trainer = ClipAdapterTrainer(clip_adapter, image_features, text_features)

    loss = nn.CrossEntropyLoss()
    parameters = (clip_adapter.vision_adapter.parameters(), clip_adapter.text_adapter.parameters())
    optimizer = torch.optim.Adam(itertools.chain(*parameters), **adam_params)

    train_model(train_loader, val_loader, clip_adapter_trainer, loss, optimizer, torch_device, epochs_num, checkpoints_path)


@hydra.main(config_path='conf', config_name='train_adapter', version_base='1.1')
def run(cfg: DictConfig) -> None:
    logging.info('Start!')
    print(OmegaConf.to_yaml(cfg))
    wandb.init(project='train_adapter', config=tp.cast(dict, OmegaConf.to_container(cfg)))
    adapter_fabric = hydra.utils.instantiate(cfg.adapter)
    train_adapter(
        cfg.clip.model_name, cfg.dataset, cfg.data.validation_size, cfg.data.batch_size,
        cfg.data.num_workers, adapter_fabric, cfg.prompting.classes, cfg.prompting.templates, cfg.data.image_features_path,
        cfg.training.adam_params, cfg.training.epochs_num, cfg.data.checkpoints_dir, cfg.meta.device, cfg.meta.random_state
    )
    logging.info('Finish!')


if __name__ == '__main__':
    run()
