import typing as tp
from pathlib import Path

import clip
import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from summer_clip.clip_model import eval_clip
from summer_clip.utils import hydra_utils
from summer_clip.utils.trainer import run_trainer, BaseTrainer
from summer_clip.clip_searcher.utils import load_labels, compute_accuracy, TensorsNumpySaver
from summer_clip.clip_searcher.cache_strategy import CacheStrategy, IndexedCacheStrategy


class ImageAttention(BaseTrainer):
    def setup_dataset(self):
        self.dataset = hydra.utils.instantiate(self.cfg.dataset)
        self.test_labels = load_labels(self.dataset).to(self.device)
        self.cache_labels: tp.Optional[torch.Tensor] = None
        if self.cfg.cache.dataset:
            cache_dataset = hydra.utils.instantiate(self.cfg.cache.dataset)
            self.cache_labels = load_labels(cache_dataset).to(self.device)
        if self.cfg.run_saves.save_labels:
            self.save_labels()

    def setup_logger(self):
        super().setup_logger()
        self.gold_labels_saver = TensorsNumpySaver(Path('./gold_labels'))
        self.cache_saver = TensorsNumpySaver(Path('./cache_ids'))
        self.preds_saver = TensorsNumpySaver(Path('./preds_ids'))

    def save_labels(self) -> None:
        self.gold_labels_saver.save_named_tensor(self.test_labels, "test_labels")
        if self.cache_labels is not None:
            self.gold_labels_saver.save_named_tensor(self.cache_labels, "cache_labels")

    def load_test_text_features(self, device: torch.device):
        clip_model, _ = clip.load(self.cfg.clip.model_name, device, jit=False)
        clip_classes = self.cfg.prompting.classes or self.dataset.classes

        test_text_features = eval_clip.zeroshot_classifier(
            clip_model, clip_classes, self.cfg.prompting.templates, device=device  # type: ignore
        )
        return test_text_features.to(device)

    def build_cache(self, cache_strategy: CacheStrategy, image_features: torch.Tensor, image_outs: torch.Tensor) \
            -> tp.Tuple[torch.Tensor, torch.Tensor, tp.Dict[str, tp.Any]]:
        if not isinstance(cache_strategy, IndexedCacheStrategy):
            cache_image_features, cache_image_outs = cache_strategy.transform(image_features, image_outs)
            return cache_image_features, cache_image_outs, {}

        samples_inds = cache_strategy.select(image_features, image_outs)
        cache_image_features, cache_image_outs = image_features[:, samples_inds], image_outs[samples_inds]
        cache_info: tp.Dict[str, tp.Any] = dict(cache_size=cache_image_outs.shape[0])

        if self.cfg.run_saves.save_cache_inds:
            cache_info['cache_inds_path'] = str(self.cache_saver.save_tensor(samples_inds))

        if self.cache_labels is not None:
            cache_labels = self.cache_labels[samples_inds]
            eval_top1, eval_top5 = compute_accuracy(cache_image_outs, cache_labels)
            cache_info.update(dict(acc1=eval_top1, acc5=eval_top5))
            if self.cfg.cache.get('replace_outs_with_golds', False):
                cache_image_outs = F.one_hot(cache_labels.long(), num_classes=cache_image_outs.shape[1]).float()
                eval_top1, eval_top5 = compute_accuracy(cache_image_outs, cache_labels)
                cache_info.update(dict(acc1_replace=eval_top1, acc5_replace=eval_top5))

        return cache_image_features, cache_image_outs, cache_info

    def setup_model(self):
        self.test_text_features = self.load_test_text_features(self.device).float()
        self.test_image_features = torch.load(self.cfg.data.image_features_path, map_location=self.device).float()

        self.origin_cache_image_features = torch.load(self.cfg.cache.image_features_path, map_location=self.device).float()
        self.origin_cache_image_outs = torch.load(self.cfg.cache.image_outs_path, map_location=self.device).float()
        self.logger.log_info(f'original-data-size: {self.origin_cache_image_outs.shape[0]}')

    def compute_clip_logits(self) -> torch.Tensor:
        norm_test_image_features = self.test_image_features / self.test_image_features.norm(dim=0, keepdim=True)
        clip_logits = 100. * norm_test_image_features.t() @ self.test_text_features
        return clip_logits

    def logits_to_preds(self, logits: torch.Tensor) -> torch.Tensor:
        _, preds = logits.max(dim=1)
        return preds

    def train_loop(self):
        clip_logits = self.compute_clip_logits()
        eval_top1, eval_top5 = compute_accuracy(clip_logits, self.test_labels)
        zeroshot_info: tp.Dict[str, tp.Any] = dict(acc1=eval_top1, acc5=eval_top5)
        if self.cfg.run_saves.save_preds:
            zeroshot_info['preds_path'] = str(self.preds_saver.save_tensor(self.logits_to_preds(clip_logits)))
        if self.cfg.run_saves.save_logits:
            zeroshot_info['logits_path'] = str(self.preds_saver.save_tensor(clip_logits))
        self.logger.log_info(dict(**zeroshot_info, type='zero_shot'))

        for cache_strategy_cfg in self.cfg.cache_strategies.values():
            for cache_strategy, cache_strategy_params in hydra_utils.instantiate_all(cache_strategy_cfg):
                cache_image_features, cache_image_outs, cache_info = self.build_cache(
                    cache_strategy, self.origin_cache_image_features, self.origin_cache_image_outs
                )
                self.logger.log_info(dict(**cache_info, cache_strategy=cache_strategy_params, type='cache_info'))
                for cache_weights_strategy, cache_weights_strategy_params in hydra_utils.instantiate_all(self.cfg.cache_weights_strategy):
                    cache_weights = cache_weights_strategy.transform(self.test_image_features, cache_image_features)
                    for cache_value_strategy, cache_value_strategy_params in hydra_utils.instantiate_all(self.cfg.cache_value_strategy):
                        cache_logits = cache_weights @ cache_value_strategy.transform(cache_image_outs)
                        for alpha in self.cfg.cache.alpha:
                            searcher_logits = clip_logits + cache_logits * alpha
                            eval_top1, eval_top5 = compute_accuracy(searcher_logits, self.test_labels)
                            searcher_info: tp.Dict[str, tp.Any] = dict(
                                cache_strategy=cache_strategy_params, cache_value_strategy=cache_value_strategy_params,
                                cache_weights_strategy=cache_weights_strategy_params, alpha=alpha,
                                acc1=eval_top1, acc5=eval_top5
                            )
                            if self.cfg.run_saves.save_preds:
                                searcher_info['preds_path'] = str(self.preds_saver.save_tensor(self.logits_to_preds(searcher_logits)))
                            self.logger.log_info_wandb(dict(**searcher_info, type='searcher_result'))


@hydra.main(config_path='../conf', config_name='image_attention', version_base='1.2')
def run(cfg: DictConfig) -> None:
    run_trainer(ImageAttention, cfg)


if __name__ == '__main__':
    run()
