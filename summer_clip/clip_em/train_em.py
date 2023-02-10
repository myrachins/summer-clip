import clip
import hydra
import torch
import joblib
import numpy as np
from omegaconf import DictConfig

from summer_clip.clip_model import eval_clip
from summer_clip.utils.trainer import run_trainer, BaseTrainer
from summer_clip.clip_searcher.utils import load_labels, compute_accuracy


def compute_np_accuracy(logits: np.ndarray, test_labels: np.ndarray):
    return compute_accuracy(torch.as_tensor(logits), torch.as_tensor(test_labels))


class ClipEM(BaseTrainer):
    def setup_logger(self):
        pass

    def setup_dataset(self):
        self.dataset = hydra.utils.instantiate(self.cfg.dataset)
        self.labels = load_labels(self.dataset).numpy()

    def load_text_features(self, device):
        """is taken from ImageAttention.load_test_text_features"""
        clip_model, _ = clip.load(self.cfg.clip.model_name, device, jit=False)
        clip_classes = self.cfg.prompting.classes or self.dataset.classes

        text_features = eval_clip.zeroshot_classifier(clip_model, clip_classes, self.cfg.prompting.templates)
        return text_features.cpu().numpy()

    def setup_model(self):
        self.text_features = self.load_text_features(self.cfg.meta.device)

        self.image_features = torch.load(self.cfg.data.image_features_path)
        self.image_features /= self.image_features.norm(dim=0, keepdim=True)
        self.image_features = self.image_features.cpu().numpy()

        self.model = hydra.utils.instantiate(
            self.cfg.em_model,
            n_components=self.text_features.shape[-1],
            means_init=self.text_features.T
        )

    def compute_clip_accuracy(self):
        clip_logits = self.image_features.T @ self.text_features
        return compute_np_accuracy(clip_logits, self.labels)

    def train_loop(self):
        clip_top1, clip_top5 = self.compute_clip_accuracy()
        print(f'Zero-shot CLIP: acc@1: {clip_top1}, acc@5: {clip_top5}')

        self.model.fit(self.image_features.T)
        em_logits = self.model.predict_proba(self.image_features.T)
        em_top1, em_top5 = compute_np_accuracy(em_logits, self.labels)
        print(f'EM-CLIP: acc@1: {em_top1}, acc@5: {em_top5}')

        joblib.dump(self.model, self.cfg.save_model.name)
        print('Model was saved!')


@hydra.main(config_path='../conf', config_name='train_em', version_base='1.1')
def run(cfg: DictConfig) -> None:
    run_trainer(ClipEM, cfg)


if __name__ == '__main__':
    run()
