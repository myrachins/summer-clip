import logging
import typing as tp

import clip
import hydra
import torch
import torch.cuda
import torch.backends.cudnn
import torch.utils
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from torch.utils.data.dataloader import DataLoader

from summer_clip.clip_adapter import train_adapter
from summer_clip.utils.trainer import set_random_state


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


def eval_clip(model_name: str, dataset_cfg: DictConfig, classes: tp.Optional[tp.List[str]], templates: tp.List[str],
              image_features_path: str, batch_size: int, num_workers: int, device: str, random_state: int):
    set_random_state(random_state)

    clip_model, _ = clip.load(model_name, device)
    dataset = hydra.utils.instantiate(dataset_cfg)
    indexed_dataset = train_adapter.NoImageIndexedDataset(dataset)
    loader = DataLoader(indexed_dataset, batch_size=batch_size, num_workers=num_workers)

    clip_classes = classes or dataset.classes
    text_features = zeroshot_classifier(clip_model, clip_classes, templates)
    image_features = torch.load(image_features_path).to(device)

    top1, top5 = train_adapter.compute_accuracy(image_features, text_features, loader)
    logging.info(f'acc@1: {top1}')
    logging.info(f'acc@5: {top5}')


@hydra.main(config_path='../conf', config_name='eval_clip', version_base='1.1')
def run(cfg: DictConfig) -> None:
    logging.info('Start!')
    print(OmegaConf.to_yaml(cfg))

    eval_clip(
        cfg.clip.model_name, cfg.dataset, cfg.prompting.classes, cfg.prompting.templates,
        cfg.eval.image_features_path, cfg.data.batch_size, cfg.data.num_workers,
        cfg.meta.device, cfg.meta.random_state
    )
    logging.info('Finish!')


if __name__ == '__main__':
    run()
