import typing as tp

import clip
import hydra
import torch
from omegaconf import DictConfig
from sklearn.decomposition import PCA

from summer_clip.clip_model import eval_clip
from summer_clip.utils.trainer import run_trainer, BaseTrainer
from summer_clip.clip_searcher.utils import load_labels, compute_accuracy


class PCATorchWrapper:
    def __init__(self, pca: PCA) -> None:
        self.pca = pca

    @staticmethod
    def _apply_method(method: tp.Callable, x: torch.Tensor) -> torch.Tensor:
        y = torch.as_tensor(method(x.cpu().numpy()))
        y = y.to(x.device)
        return y

    def fit_transform(self, x: torch.Tensor) -> torch.Tensor:
        return self._apply_method(self.pca.fit_transform, x)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return self._apply_method(self.pca.transform, x)


class ClassProjector(BaseTrainer):
    def setup_dataset(self):
        self.dataset = hydra.utils.instantiate(self.cfg.dataset)
        self.test_labels = load_labels(self.dataset).to(self.cfg.meta.device)

    def load_test_text_features(self, device: torch.device):
        clip_model, _ = clip.load(self.cfg.clip.model_name, device, jit=False)
        clip_classes = self.cfg.prompting.classes or self.dataset.classes

        test_text_features = eval_clip.zeroshot_classifier(clip_model, clip_classes, self.cfg.prompting.templates)
        return test_text_features.to(device)

    def setup_model(self):
        device = torch.device(self.cfg.meta.device)
        self.test_text_features = self.load_test_text_features(device)
        self.logger.log_info(f'text features shape: {self.test_text_features.shape}')
        self.test_image_features = torch.load(self.cfg.data.image_features_path).to(device)
        self.test_image_features /= self.test_image_features.norm(dim=0, keepdim=True)
        self.logger.log_info(f'image features shape: {self.test_image_features.shape}')

    @staticmethod
    def compute_clip_logits(image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        image_features = image_features / image_features.norm(dim=0, keepdim=True)
        text_features = text_features / text_features.norm(dim=0, keepdim=True)
        clip_logits = 100. * image_features.t() @ text_features
        return clip_logits

    def train_loop(self):
        clip_logits = self.compute_clip_logits(self.test_image_features, self.test_text_features)
        eval_top1, eval_top5 = compute_accuracy(clip_logits, self.test_labels)
        self.logger.log_info(f'zero-shot clip: acc@1={eval_top1}, acc@5={eval_top5}')

        for n_components in self.cfg.pca.n_components:
            pca = PCATorchWrapper(PCA(n_components=n_components))
            test_text_features = pca.fit_transform(self.test_text_features.t()).t()
            test_image_features = pca.transform(self.test_image_features.t()).t()
            pca_logits = self.compute_clip_logits(test_image_features, test_text_features)
            eval_top1, eval_top5 = compute_accuracy(pca_logits, self.test_labels)
            self.logger.log_info(dict(n_components=n_components, acc1=eval_top1, acc5=eval_top5))


@hydra.main(config_path='../conf', config_name='class_projector', version_base='1.2')
def run(cfg: DictConfig) -> None:
    run_trainer(ClassProjector, cfg)


if __name__ == '__main__':
    run()
