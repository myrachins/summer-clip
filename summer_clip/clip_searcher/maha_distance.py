import torch
import hydra
from omegaconf import DictConfig

from summer_clip.utils.trainer import run_trainer
from summer_clip.clip_searcher.utils import compute_accuracy
from summer_clip.clip_searcher.class_projector import ClassProjector


class MahaDistance(ClassProjector):
    def setup_model(self):
        super().setup_model()
        self.cache_image_features = torch.load(self.cfg.cache.image_features_path).to(self.cfg.meta.device)
        self.cache_image_features /= self.cache_image_features.norm(dim=0, keepdim=True)
        self.logger.log_info(f'cache image features shape: {self.test_image_features.shape}')

    def train_loop(self):
        clip_logits = self.compute_clip_logits(self.test_image_features, self.test_text_features)
        eval_top1, eval_top5 = compute_accuracy(clip_logits, self.test_labels)
        self.logger.log_info(f'zero-shot clip: acc@1={eval_top1}, acc@5={eval_top5}')

        test_image_features = self.test_image_features.t().float()
        cache_image_features = self.cache_image_features.t().float()
        test_text_features = self.test_text_features.t().float()

        image_text_features = torch.cat([cache_image_features, test_text_features], dim=0)
        cov_mat = torch.cov(image_text_features.t()) * (image_text_features.shape[0] - 1)
        # print('cov_mat', cov_mat[0][0], cov_mat[0][1], cov_mat.max(), cov_mat.min())
        # cov_mat = torch.eye(cache_image_features.shape[1]).to(self.cfg.meta.device)  # check for being clip
        inv_cov_mat = torch.linalg.inv(cov_mat)
        # print('inv_cov_mat', inv_cov_mat[0][0], inv_cov_mat[0][1], inv_cov_mat.max(), inv_cov_mat.min())
        inv_cov_mat = inv_cov_mat.unsqueeze(dim=0).expand(test_image_features.shape[0], -1, -1)

        image_broad = test_image_features.unsqueeze(dim=1)
        text_broad = test_text_features.unsqueeze(dim=0)
        diff = image_broad - text_broad
        maha_1 = torch.bmm(diff, inv_cov_mat)
        maha_logits = (maha_1 * diff).sum(dim=2)

        eval_top1, eval_top5 = compute_accuracy(-maha_logits, self.test_labels)
        self.logger.log_info(f'Maha clip: acc@1={eval_top1}, acc@5={eval_top5}')


@hydra.main(config_path='../conf', config_name='maha_distance', version_base='1.2')
def run(cfg: DictConfig) -> None:
    run_trainer(MahaDistance, cfg)


if __name__ == '__main__':
    run()
