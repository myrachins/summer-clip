import os

import clip
import hydra
import omegaconf
import torchvision.transforms as transforms
from omegaconf import DictConfig

from summer_clip.tip_adapter.datasets import build_dataset
from summer_clip.tip_adapter.datasets.utils import build_data_loader
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

        print("Preparing dataset.")
        dataset = build_dataset(self.cfg['dataset'], self.cfg['root_path'], self.cfg['shots'])

        val_loader = build_data_loader(data_source=dataset.val, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)

        train_tranform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])

        train_loader_cache = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=False)
        
        print("\nGetting textual features as CLIP's classifier.")
        self.clip_weights = tip_utils.clip_classifier(dataset.classnames, dataset.template, clip_model)

        # Construct the cache model by few-shot training set
        print("\nConstructing cache model by few-shot visual features and labels.")
        self.cache_keys, self.cache_values = tip_utils.build_cache_model(self.cfg, clip_model, train_loader_cache)

        # Pre-load val features
        print("\nLoading visual features and labels from val set.")
        self.val_features, self.val_labels = tip_utils.pre_load_features(self.cfg, "val", clip_model, val_loader)

        # Pre-load test features
        print("\nLoading visual features and labels from test set.")
        self.test_features, self.test_labels = tip_utils.pre_load_features(self.cfg, "test", clip_model, test_loader)

    def train_loop(self):
        # Zero-shot CLIP
        clip_logits = 100. * self.test_features @ self.clip_weights
        acc = tip_utils.cls_acc(clip_logits, self.test_labels)
        self.logger.log_info(f"**** Zero-shot CLIP's test accuracy: {acc}. ****")

        # Tip-Adapter
        beta, alpha = self.cfg['init_beta'], self.cfg['init_alpha']

        affinity = self.test_features @ self.cache_keys
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ self.cache_values

        tip_logits = clip_logits + cache_logits * alpha
        acc = tip_utils.cls_acc(tip_logits, self.test_labels)
        self.logger.log_info(f"**** Tip-Adapter's test accuracy: {acc}. ****")

        # Search Hyperparameters
        tip_utils.search_hp(
            self.cfg, self.cache_keys, self.cache_values, self.test_features, self.test_labels, self.clip_weights
        )


@hydra.main(config_path='../conf', config_name='tip_adapter', version_base='1.1')
def run(cfg: DictConfig) -> None:
    run_trainer(TipAdapterTrainer, cfg)


if __name__ == '__main__':
    run()
