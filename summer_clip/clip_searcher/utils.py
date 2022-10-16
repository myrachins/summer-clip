from pathlib import Path

import torch
import numpy as np
from torch.utils.data.dataset import Dataset

from summer_clip.clip_adapter import train_adapter


def load_labels(dataset: Dataset) -> torch.Tensor:
    labels = [label for _, label in dataset]  # type: ignore
    return torch.IntTensor(labels)


def compute_accuracy(outputs, target, topk=(1, 5)):
    acc_ks = train_adapter.accuracy(outputs, target, topk)

    def transform_acc(acc):
        return 100. * acc / target.shape[0]

    return [transform_acc(acc) for acc in acc_ks]


class FilesNamesManager:
    def __init__(self, dir_path: Path, files_ext: str) -> None:
        self.dir_path = dir_path.resolve()
        self.files_ext = files_ext
        self.dir_path.mkdir(parents=True, exist_ok=True)
        self.counter = 0

    def next_path(self) -> Path:
        new_path = self.get_path(str(self.counter))
        self.counter += 1
        return new_path

    def get_path(self, file_name: str) -> Path:
        return self.dir_path / f'{file_name}{self.files_ext}'


class TensorsNumpySaver:
    def __init__(self, dir_path: Path) -> None:
        self.files_names_manager = FilesNamesManager(dir_path, files_ext='.npy')

    def save_tensor(self, tensor: torch.Tensor) -> Path:
        tensor_path = self.files_names_manager.next_path()
        return self.save_named_tensor(tensor, tensor_path.with_suffix('').name)

    def save_named_tensor(self, tensor: torch.Tensor, file_name: str) -> Path:
        tensor_np = tensor.cpu().numpy()
        tensor_path = self.files_names_manager.get_path(file_name)
        np.save(tensor_path, tensor_np)
        return tensor_path
