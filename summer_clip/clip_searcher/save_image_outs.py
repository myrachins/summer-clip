import hydra
import torch
import clip
from omegaconf import DictConfig

from summer_clip.clip_model import eval_clip
from summer_clip.utils.trainer import BaseTrainer
from summer_clip.utils.trainer import run_trainer


class SaveImageOuts(BaseTrainer):
    def setup_dataset(self):
        self.dataset = hydra.utils.instantiate(self.cfg.dataset)

    def setup_model(self):
        device = torch.device(self.cfg.meta.device)
        clip_model, _ = clip.load(self.cfg.clip.model_name, device, jit=False)
        clip_classes = self.cfg.prompting.classes or self.dataset.classes
        self.text_features = eval_clip.zeroshot_classifier(clip_model, clip_classes, self.cfg.prompting.templates).to(device)
        self.image_features = torch.load(self.cfg.data.image_features_path).to(device)

    def train_loop(self):
        print('Computing outputs...')
        image_outs = self.image_features.t() @ self.text_features
        print('Saving outputs...')
        torch.save(image_outs, self.cfg.data.output_image_outs)
        print('Saved!')


@hydra.main(config_path='../conf', config_name='save_image_outs', version_base='1.1')
def run(cfg: DictConfig) -> None:
    run_trainer(SaveImageOuts, cfg)


if __name__ == '__main__':
    run()
