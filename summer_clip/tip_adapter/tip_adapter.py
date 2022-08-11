import os
import typing as tp

import clip
import torch
import hydra
import omegaconf
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from omegaconf import DictConfig

from summer_clip.tip_adapter.imagenet import ImageNet
from summer_clip.utils.trainer import BaseTrainer
from summer_clip.utils.trainer import run_trainer
from summer_clip.tip_adapter import utils as tip_utils


class TipAdapterTrainer(BaseTrainer):
    def setup_model(self):
        cache_dir = os.path.join('./caches', self.cfg['dataset'])
        os.makedirs(cache_dir, exist_ok=True)
        with omegaconf.open_dict(self.cfg):
            self.cfg['cache_dir'] = cache_dir

        clip_model, preprocess = clip.load(self.cfg['backbone'], self.cfg.meta.device)
        clip_model.eval()

        print("Preparing ImageNet dataset.")
        imagenet = ImageNet(self.cfg['root_path'], self.cfg['shots'], preprocess)
        test_loader = DataLoader(imagenet.test, batch_size=64, num_workers=8, shuffle=False)
        train_loader_cache = DataLoader(imagenet.train, batch_size=256, num_workers=8, shuffle=False)
        
        print("Getting textual features as CLIP's classifier.")
        self.clip_weights = tip_utils.clip_classifier(imagenet.classnames, imagenet.template, clip_model)

        print("\nConstructing cache model by few-shot visual features and labels.")
        self.cache_keys, self.cache_values = tip_utils.build_cache_model(self.cfg, clip_model, train_loader_cache)

        print("\nLoading visual features and labels from test set.")
        self.test_features, self.test_labels = tip_utils.pre_load_features(self.cfg, "test", clip_model, test_loader)

    def train_loop(self):
        # Zero-shot CLIP
        clip_logits = 100. * self.test_features @ self.clip_weights
        acc = tip_utils.cls_acc(clip_logits, self.test_labels)
        print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))

        # Tip-Adapter
        beta, alpha = self.cfg['init_beta'], self.cfg['init_alpha']

        affinity = self.test_features @ self.cache_keys
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ self.cache_values

        tip_logits = clip_logits + cache_logits * alpha
        acc = tip_utils.cls_acc(tip_logits, self.test_labels)
        print("**** Tip-Adapter's test accuracy: {:.2f}. ****\n".format(acc))

        # Search Hyperparameters
        tip_utils.search_hp(
            self.cfg, self.cache_keys, self.cache_values, self.test_features, self.test_labels, self.clip_weights
        )


@hydra.main(config_path='../conf', config_name='tip_adapter', version_base='1.1')
def run(cfg: DictConfig) -> None:
    run_trainer(TipAdapterTrainer, cfg)


if __name__ == '__main__':
    run()
