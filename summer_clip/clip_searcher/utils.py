import torch
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
