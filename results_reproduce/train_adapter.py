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
from torch import nn
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from torch.utils.data.dataset import random_split
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from results_reproduce import zero_shot


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
    def __init__(self, clip_adapter: ClipAdapter, classes: tp.List[str], templates: tp.List[str]) -> None:
        super().__init__()
        self.clip_adapter = clip_adapter
        self.classes = classes
        self.templates = templates

        zeroshot_weights = zero_shot.zeroshot_classifier(clip_adapter.clip_model, classes, templates)
        self.zeroshot_weights = nn.parameter.Parameter(zeroshot_weights, requires_grad=False)

    def forward(self, image, label):
        with torch.no_grad():
            image_features = self.clip_adapter.clip_model.encode_image(image)
            text_features = self.zeroshot_weights[:, label].t()

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


# https://github.com/openai/CLIP/issues/57
def convert_model_to_fp32(model: nn.Module) -> None:
    for param in model.parameters():
        param.data = param.data.float()
        if param.requires_grad and param.grad is not None:
            param.grad.data = param.grad.data.float()


def train_epoch(loader: DataLoader, model: ClipAdapterTrainer, loss: nn.CrossEntropyLoss, optimizer: torch.optim.Optimizer,
                device: torch.device, summary_writer: SummaryWriter, tb_loss_step: int):
    model.train()
    model = model.to(device)
    convert_model_to_fp32(model)
    epoch_loss = 0.

    for images, labels in tqdm(loader):
        model.zero_grad()
        images = images.to(device)
        labels = labels.to(device)

        logits_per_image, logits_per_text = model(images, labels)
        dummy_labels = torch.arange(len(labels), device=device)
        image_loss = loss(logits_per_image, dummy_labels)
        text_loss = loss(logits_per_text, dummy_labels)
        agg_loss = (image_loss + text_loss) / 2

        summary_writer.add_scalar('loss-train-image', image_loss.item(), tb_loss_step)
        summary_writer.add_scalar('loss-train-text', text_loss.item(), tb_loss_step)
        summary_writer.add_scalar('loss-train-agg', agg_loss.item(), tb_loss_step)
        tb_loss_step += 1
        epoch_loss += agg_loss.item()

        agg_loss.backward()
        # convert_models_to_fp32(model)
        optimizer.step()
        # clip.model.convert_weights(model)

    return model, epoch_loss, optimizer, tb_loss_step


def eval_model(loader: DataLoader, model: ClipAdapterTrainer) -> tp.Tuple[float, float]:
    # recompute zeroshot_weights since we need them with adapter performed (can be optimized)
    zeroshot_weights = zero_shot.zeroshot_classifier(model.clip_adapter, model.classes, model.templates)
    return zero_shot.compute_accuracy(model.clip_adapter, zeroshot_weights, loader)


def save_epoch_model(model: ClipAdapter, optimizer: torch.optim.Optimizer, tb_loss_step: int, epoch_num: int, checkpoints_dir: Path):
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
    save_data(tb_loss_step, 'tb_loss_step')


def train_model(train_loader: DataLoader, val_loader: DataLoader, model: ClipAdapterTrainer, loss: nn.CrossEntropyLoss,
                optimizer: torch.optim.Optimizer, device: torch.device, epochs_num: int, checkpoints_dir: Path):
    log_dir = checkpoints_dir / 'tb_runs'
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_writer = SummaryWriter(log_dir)
    tb_loss_step = 0

    for epoch_num in range(1, epochs_num + 1):
        print(f'Running epoch {epoch_num}...')
        model, epoch_loss, optimizer, tb_loss_step = train_epoch(train_loader, model, loss, optimizer, device, summary_writer, tb_loss_step)
        print('Evaluating model on train...')
        eval_top1, eval_top5 = eval_model(train_loader, model)
        summary_writer.add_scalar('train-epoch-sum-loss', epoch_loss, epoch_num)
        summary_writer.add_scalar('train-epoch-acc@1', eval_top1, epoch_num)
        summary_writer.add_scalar('train-epoch-acc@5', eval_top5, epoch_num)
        logging.info(f'{epoch_num=}, train-acc@1: {eval_top1}')
        logging.info(f'{epoch_num=}, train-acc@5: {eval_top5}')
        print('Evaluating model on validation...')
        eval_top1, eval_top5 = eval_model(val_loader, model)
        summary_writer.add_scalar('val-epoch-acc@1', eval_top1, epoch_num)
        summary_writer.add_scalar('val-epoch-acc@5', eval_top5, epoch_num)
        logging.info(f'{epoch_num=}, val-acc@1: {eval_top1}')
        logging.info(f'{epoch_num=}, val-acc@5: {eval_top5}')
        print(f'Saving checkpoint after {epoch_num} epoch...')
        save_epoch_model(model.clip_adapter, optimizer, tb_loss_step, epoch_num, checkpoints_dir)


def train_val_split(dataset, validation_size: float):
    val_size = int(len(dataset) * validation_size)
    train_size = len(dataset) - val_size
    return random_split(dataset, [train_size, val_size])


def train_adapter(model_name: str, dataset_name: str, validation_size: float, learning_rate: float, batch_size: int, num_workers: int,
                  adapter_fabric: ClipAdapterFabric, classes: tp.Optional[tp.List[str]], templates: tp.List[str],
                  epochs_num: int, checkpoints_dir: str, device: str, random_state: int) -> None:
    zero_shot.set_random_state(random_state)
    torch_device = torch.device(device)
    checkpoints_path = Path(checkpoints_dir)

    clip_model, preprocess = clip.load(model_name, device, jit=False)
    dataset = zero_shot.get_dataset(dataset_name, preprocess)
    train_dataset, val_dataset = train_val_split(dataset, validation_size)
    logging.info(f'train-size={len(train_dataset)}, val-size={len(val_dataset)}')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

    clip_adapter = adapter_fabric.create_adapter(clip_model)
    clip_classes = classes or dataset.classes
    clip_adapter_trainer = ClipAdapterTrainer(clip_adapter, clip_classes, templates)

    loss = nn.CrossEntropyLoss()
    parameters = (clip_adapter.vision_adapter.parameters(), clip_adapter.text_adapter.parameters())
    optimizer = torch.optim.Adam(itertools.chain(*parameters), lr=learning_rate, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)

    train_model(train_loader, val_loader, clip_adapter_trainer, loss, optimizer, torch_device, epochs_num, checkpoints_path)


@hydra.main(config_path='conf', config_name='train_adapter', version_base='1.1')
def run(cfg: DictConfig) -> None:
    logging.info('Start!')
    print(OmegaConf.to_yaml(cfg))
    adapter_fabric = hydra.utils.instantiate(cfg.adapter)
    train_adapter(
        cfg.clip.model_name, cfg.data.dataset_name, cfg.data.validation_size, cfg.training.learning_rate, cfg.training.batch_size,
        cfg.data.num_workers, adapter_fabric, cfg.prompting.classes, cfg.prompting.templates, cfg.training.epochs_num,
        cfg.data.checkpoints_dir, cfg.meta.device, cfg.meta.random_state
    )
    logging.info('Finish!')


if __name__ == '__main__':
    run()
