import json
from pathlib import Path

import omegaconf
import hydra
import torch
from omegaconf import DictConfig

from summer_clip.utils import hydra_utils
from summer_clip.utils.trainer import run_trainer
from summer_clip.clip_searcher.image_attention import ImageAttention


class ClassDistribution(ImageAttention):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        with omegaconf.open_dict(self.cfg.cache):
            self.cfg.cache['replace_outs_with_golds'] = True

    def train_loop(self):
        for cache_strategy, cache_strategy_params in hydra_utils.instantiate_all(self.cfg.cache_strategy):
            _, cache_image_outs = self.build_cache(
                cache_strategy, self.origin_cache_image_features, self.origin_cache_image_outs
            )
            _, cache_image_labels = cache_image_outs.max(dim=1)
            cache_image_labels_path = Path("selected_cache") / f'{json.dumps(cache_strategy_params)}.pt'
            cache_image_labels_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(cache_image_labels.cpu(), cache_image_labels_path)

        torch.save(self.test_labels.cpu(), "test_labels.pt")
        assert self.cache_labels is not None, "cache_labels are none"
        torch.save(self.cache_labels.cpu(), "cache_labels.pt")


@hydra.main(config_path='../conf', config_name='image_attention', version_base='1.2')
def run(cfg: DictConfig) -> None:
    run_trainer(ClassDistribution, cfg)


if __name__ == '__main__':
    run()
