import os
import importlib

import fire
import clip
import torch
import torch.utils
from tqdm import tqdm


def get_dataset(dataset_name, preprocess):
    root = os.path.expanduser("~/.cache")
    if dataset_name == 'CIFAR100':
        from torchvision.datasets import CIFAR100
        return CIFAR100(root=root, train=False, transform=preprocess)
    elif dataset_name == 'ImageNet':
        from torchvision.datasets import ImageNet
        return ImageNet(root=root, split='val', transform=preprocess)
    elif dataset_name == 'CIFAR10':
        from torchvision.datasets import CIFAR10
        return CIFAR10(root=root, train=False, transform=preprocess)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def load_promts(dataset_name):
    promts_mapping = {'CIFAR100': 'cifar100', 'CIFAR10': 'cifar10'}
    if dataset_name not in promts_mapping:
        raise ValueError("Unsupported dataset for promts: {dataset_name}")

    module_name = promts_mapping[dataset_name]
    module = importlib.import_module(f'results_reproduce.promts.{module_name}')
    return module.classes, module.templates


def zeroshot_classifier(model, classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
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


def print_accuracy(model, zeroshot_weights, loader):
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


    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100

    print(f"Top-1 accuracy: {top1:.2f}")
    print(f"Top-5 accuracy: {top5:.2f}")


def run(model_name: str = 'ViT-L/14@336px', dataset_name: str = 'CIFAR100', batch_size: int = 32, num_workers: int = 2, device: str ='cuda'):
    model, preprocess = clip.load(model_name, device)
    dataset = get_dataset(dataset_name, preprocess)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    classes, templates = load_promts(dataset_name)
    zeroshot_weights = zeroshot_classifier(model, classes, templates)
    print_accuracy(model, zeroshot_weights, loader)


if __name__ == '__main__':
    fire.Fire(run)
