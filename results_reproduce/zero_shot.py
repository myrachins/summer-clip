import os
import importlib
import random

import fire
import clip
import torch
import torch.cuda
import torch.backends.cudnn
import torch.utils
import numpy as np
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader


def get_dataset(dataset_name, preprocess):
    root = os.path.expanduser("~/.cache")
    if dataset_name == 'CIFAR100-test':
        from torchvision.datasets import CIFAR100
        return CIFAR100(root=root, train=False, transform=preprocess)
    elif dataset_name == 'CIFAR100-train':
        from torchvision.datasets import CIFAR100
        return CIFAR100(root=root, train=True, transform=preprocess)
    elif dataset_name == 'CIFAR10-test':
        from torchvision.datasets import CIFAR10
        return CIFAR10(root=root, train=False, transform=preprocess)
    elif dataset_name == 'CIFAR10-train':
        from torchvision.datasets import CIFAR10
        return CIFAR10(root=root, train=True, transform=preprocess)
    elif dataset_name == 'ImageNetV2':
        from imagenetv2_pytorch import ImageNetV2Dataset
        return ImageNetV2Dataset(location=root, transform=preprocess)
    elif dataset_name == 'ImageNet-val':
        from torchvision.datasets import ImageNet
        return ImageNet(f'{root}/ImageNet', split='val', transform=preprocess)
    elif dataset_name == 'ImageNet-train':
        from torchvision.datasets import ImageNet
        return ImageNet(f'{root}/ImageNet', split='train', transform=preprocess)
    elif dataset_name == 'MNIST-test':
        from torchvision.datasets import MNIST
        return MNIST(root=root, train=False, transform=preprocess)
    elif dataset_name == 'MNIST-train':
        from torchvision.datasets import MNIST
        return MNIST(root=root, train=True, transform=preprocess)

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def load_promts(dataset_name: str):
    promts_mapping = {
        'CIFAR100-test': 'cifar100', 'CIFAR100-train': 'cifar100',
        'CIFAR10-test': 'cifar10', 'CIFAR10-train': 'cifar10',
        'ImageNet-val': 'imagenet', 'ImageNetV2': 'imagenet',
        'MNIST-test': 'mnist', 'MNIST-train': 'mnist'
    }
    if dataset_name not in promts_mapping:
        raise ValueError("Unsupported dataset for promts: {dataset_name}")

    module_name = promts_mapping[dataset_name]
    module = importlib.import_module(f'results_reproduce.promts.{module_name}')
    return module.classes, module.templates


def zeroshot_classifier(model, classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates]
            texts = clip.tokenize(texts).cuda()
            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def compute_accuracy(model, zeroshot_weights, loader):
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for i, (images, target) in enumerate(tqdm(loader)):
            images = images.cuda()
            target = target.cuda()

            # predict
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100. * image_features @ zeroshot_weights

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    if n <= 0:
        return np.nan, np.nan

    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100

    return top1, top5


def set_random_state(random_state: int):
    os.environ['PYTHONHASHSEED'] = str(random_state)
    random.seed(random_state)
    np.random.seed(random_state)

    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run(model_name: str = 'ViT-L/14@336px', dataset_name: str = 'CIFAR100', batch_size: int = 32, num_workers: int = 2,
        device: str = 'cuda', random_state: int = 42):
    print(f'{model_name=}, {dataset_name=}, {batch_size=}, {num_workers=}, {device=}')
    set_random_state(random_state)
    model, preprocess = clip.load(model_name, device)
    dataset = get_dataset(dataset_name, preprocess)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    classes, templates = load_promts(dataset_name)
    zeroshot_weights = zeroshot_classifier(model, classes, templates)

    top1, top5 = compute_accuracy(model, zeroshot_weights, loader)
    print(f"Top-1 accuracy: {top1:.2f}")
    print(f"Top-5 accuracy: {top5:.2f}")


if __name__ == '__main__':
    fire.Fire(run)
