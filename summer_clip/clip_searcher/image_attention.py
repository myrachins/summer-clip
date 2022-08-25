import itertools
import typing as tp

import clip
import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from summer_clip.clip_model import eval_clip
from summer_clip.utils.trainer import run_trainer, BaseTrainer
from summer_clip.clip_searcher.utils import load_labels, compute_accuracy
from summer_clip.clip_searcher.cache_strategy import CacheStrategy, IndexedCacheStrategy
from summer_clip.clip_searcher.cache_value_strategy import CacheValueStrategy


def make_hard_cache(cache_outs: torch.Tensor) -> torch.Tensor:
    _, labels_ids = cache_outs.max(dim=1)
    return F.one_hot(labels_ids, num_classes=cache_outs.shape[1]).half()


class ImageAttention(BaseTrainer):
    def setup_dataset(self):
        self.dataset = hydra.utils.instantiate(self.cfg.dataset)
        self.test_labels = load_labels(self.dataset).to(self.cfg.meta.device)
        self.cache_labels: tp.Optional[torch.Tensor] = None
        if self.cfg.cache.dataset:
            cache_dataset = hydra.utils.instantiate(self.cfg.cache.dataset)
            self.cache_labels = load_labels(cache_dataset).to(self.cfg.meta.device)

    def load_test_text_features(self, device: torch.device):
        clip_model, _ = clip.load(self.cfg.clip.model_name, device, jit=False)
        clip_classes = self.cfg.prompting.classes or self.dataset.classes

        test_text_features = eval_clip.zeroshot_classifier(clip_model, clip_classes, self.cfg.prompting.templates)
        return test_text_features.to(device)

    def build_cache(self, cache_strategy: CacheStrategy, image_features: torch.Tensor, image_outs: torch.Tensor) \
            -> tp.Tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(cache_strategy, IndexedCacheStrategy):
            return cache_strategy.transform(image_features, image_outs)

        samples_inds = cache_strategy.select(image_features, image_outs)
        cache_image_features, cache_image_outs = image_features[:, samples_inds], image_outs[samples_inds]

        if self.cache_labels is not None:
            cache_labels = self.cache_labels[samples_inds]
            eval_top1, eval_top5 = compute_accuracy(cache_image_outs, cache_labels)
            self.logger.log_info(f'internal cache: acc@1={eval_top1}, acc@5={eval_top5}')
            if self.cfg.cache.get('replace_outs_with_golds', False):
                cache_image_outs = F.one_hot(cache_labels.long(), num_classes=cache_image_outs.shape[1]).half()
                eval_top1, eval_top5 = compute_accuracy(cache_image_outs, cache_labels)
                self.logger.log_info(f'internal cache (after replace): acc@1={eval_top1}, acc@5={eval_top5}')

        return cache_image_features, cache_image_outs

    def setup_model(self):
        device = torch.device(self.cfg.meta.device)

        self.test_text_features = self.load_test_text_features(device)
        self.test_image_features = torch.load(self.cfg.data.image_features_path).to(device)
        self.test_image_features /= self.test_image_features.norm(dim=0, keepdim=True)

        cache_image_features = torch.load(self.cfg.cache.image_features_path).to(device)
        cache_image_outs = torch.load(self.cfg.cache.image_outs_path).to(device)
        self.logger.log_info(f'original-data-size: {cache_image_outs.shape[0]}')
        cache_strategy: CacheStrategy = hydra.utils.instantiate(self.cfg.cache_strategy)
        self.cache_image_features, self.cache_image_outs = self.build_cache(cache_strategy, cache_image_features, cache_image_outs)
        self.cache_image_features /= self.cache_image_features.norm(dim=0, keepdim=True)
        self.logger.log_info(f'cache-size: {self.cache_image_outs.shape[0]}')

    def train_loop(self):
        clip_logits = 100. * self.test_image_features.t() @ self.test_text_features
        eval_top1, eval_top5 = compute_accuracy(clip_logits, self.test_labels)
        self.logger.log_info(f'zero-shot clip: acc@1={eval_top1}, acc@5={eval_top5}')

        cache_weights = self.test_image_features.t() @ self.cache_image_features
        cache_value_strategy: CacheValueStrategy = hydra.utils.instantiate(self.cfg.cache_value_strategy)
        cache_values = cache_value_strategy.transform(self.cache_image_outs)

        for beta, alpha in itertools.product(self.cfg.cache.beta, self.cfg.cache.alpha):
            cache_logits = (-1 * beta * (1 - cache_weights)).exp() @ cache_values
            searcher_logits = clip_logits + cache_logits * alpha
            eval_top1, eval_top5 = compute_accuracy(searcher_logits, self.test_labels)
            self.logger.log_info(f'clip-searcher ({beta=}, {alpha=}): acc@1={eval_top1}, acc@5={eval_top5}')


@hydra.main(config_path='../conf', config_name='image_attention', version_base='1.2')
def run(cfg: DictConfig) -> None:
    run_trainer(ImageAttention, cfg)


if __name__ == '__main__':
    run()
