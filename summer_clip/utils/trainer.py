import os
import random

import torch
import torch.cuda
import torch.backends.cudnn
import numpy as np
import omegaconf
from omegaconf import DictConfig, OmegaConf

from summer_clip.utils import log_utils


class BaseTrainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def to(self, device):
        self.model = self.model.to(device)  # type: ignore

    def train_mode(self):
        self.model.train()  # type: ignore

    def eval_mode(self):
        self.model.eval()  # type: ignore

    def setup_device(self):
        with omegaconf.open_dict(self.cfg):
            if self.cfg.meta.device is None:
                self.cfg.meta.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.cfg.meta.device)

    def setup_logger(self):
        config_for_logger = OmegaConf.to_container(self.cfg)
        config_for_logger["PID"] = os.getpid()  # type: ignore
        exp_logger = log_utils.WandbLogger(
            project=self.cfg.exp.project,
            name=self.cfg.exp.name,
            # dir=self.cfg.exp.root,
            # tags=tuple(self.cfg.exp.tags) if self.cfg.exp.tags else None,
            # notes=self.cfg.exp.notes,
            config=config_for_logger,
        )
        self.run_dir = exp_logger.run_dir
        console_logger = log_utils.ConsoleLogger(self.cfg.exp.name)
        self.logger = log_utils.LoggingManager(exp_logger, console_logger)

    def setup_dataset(self):
        pass

    def setup_loaders(self):
        pass

    def setup_model(self):
        pass

    def setup_optimizer(self):
        pass

    def setup_scheduler(self):
        pass

    def setup_loss(self):
        pass

    def setup(self):
        self.setup_device()
        self.setup_logger()
        self.setup_dataset()
        self.setup_loaders()
        self.setup_model()
        self.setup_optimizer()
        self.setup_scheduler()
        self.setup_loss()

    def compute_metrics(self, epoch_num, epoch_info):
        pass

    def train_epoch(self, epoch_num, epoch_info):
        return epoch_info

    def validation_epoch(self, epoch_num, epoch_info):
        return epoch_info

    def save_epoch_model(self, epoch_num):
        pass

    def train_loop(self):
        training_time_log = log_utils.TimeLog(
            self.logger, self.cfg.training.epochs_num + 1, event="training"
        )
        for epoch_num in range(1, self.cfg.training.epochs_num + 1):
            epoch_info = log_utils.StreamingMeans()
            self.train_mode()
            with log_utils.Timer(epoch_info, "epoch_train"):
                epoch_info = self.train_epoch(epoch_num, epoch_info)

            self.eval_mode()
            with log_utils.Timer(epoch_info, "epoch_val"):
                epoch_info = self.validation_epoch(epoch_num, epoch_info)

            if epoch_num % self.cfg.log.calculate_every == 0:
                self.compute_metrics(epoch_num, epoch_info)

            self.logger.log_epoch(epoch_num, epoch_info)

            self.save_epoch_model(epoch_num)

            training_time_log.now(epoch_num)
        training_time_log.end()


def set_random_state(random_state: int) -> None:
    os.environ['PYTHONHASHSEED'] = str(random_state)
    random.seed(random_state)
    np.random.seed(random_state)

    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run_trainer(trainer_cls: type, cfg: DictConfig) -> BaseTrainer:
    print(OmegaConf.to_yaml(cfg))
    set_random_state(cfg.meta.random_state)

    trainer = trainer_cls(cfg)
    trainer.setup()
    trainer.train_loop()

    return trainer
