import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from summer_clip.utils.trainer import BaseTrainer
from summer_clip.utils.trainer import run_trainer


class SaveImageLabels(BaseTrainer):
    def setup_dataset(self):
        self.dataset = hydra.utils.instantiate(self.cfg.dataset)

    def train_loop(self):
        test_labels = torch.LongTensor([label for _, label in self.dataset])
        test_labels = F.one_hot(test_labels, num_classes=len(self.dataset.classes)).float()
        torch.save(test_labels, self.cfg.data.output_image_labels)


@hydra.main(config_path='../conf', config_name='save_image_labels', version_base='1.1')
def run(cfg: DictConfig) -> None:
    run_trainer(SaveImageLabels, cfg)


if __name__ == '__main__':
    run()
