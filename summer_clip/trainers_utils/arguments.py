import os
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from omegaconf import OmegaConf, MISSING
import utils
import models
import optims
import datasets


args = utils.ClassRegistry()


@args.add_to_registry("exp")
@dataclass
class ExperimentArgs:
    config_dir: str = MISSING
    config: str = MISSING
    project: str = "hifi++"
    name: str = MISSING
    seed: int = 1
    cudnn_benchmark_off: bool = False
    root: str = os.getenv("EXP_ROOT", ".")
    notes: str = "empty notes"
    tags: Optional[Tuple[str]] = None


@args.add_to_registry("training")
@dataclass
class TrainingArgs:
    trainer: str = MISSING
    num_epochs: int = MISSING
    num_iters: Optional[int] = None
    num_val_iters: Optional[int] = None
    batch_size: int = MISSING
    val_batch_size: int = MISSING
    device: str = MISSING


@args.add_to_registry("prior")
@dataclass
class PriorArgs:
    dist: str = MISSING
    latent_dim: int = MISSING


DistPriorArgs = datasets.priors.make_dataclass_from_args("DistPriorArgs")
args.add_to_registry("dist_prior")(DistPriorArgs)


@args.add_to_registry("log")
@dataclass
class LoggingArgs:
    log_every: int = MISSING
    disc_logging_off: bool = False
    log_complexity: bool = False
    calculate_every: int = MISSING
    small_val_size: int = MISSING
    val_size: int = MISSING


@args.add_to_registry("data")
@dataclass
class DataArgs:
    name: str = MISSING
    loader: str = MISSING
    root: str = MISSING
    num_workers: int = MISSING


DatasetArgs = datasets.datasets.make_dataclass_from_args("DatasetArgs")
args.add_to_registry("dataset")(DatasetArgs)

LoaderArgs = datasets.loaders.make_dataclass_from_args("LoaderArgs")
args.add_to_registry("loader")(LoaderArgs)


@args.add_to_registry("gen")
@dataclass
class GenArgs:
    model: str = MISSING
    optim: Optional[str] = None
    loss_funcs: List[str] = MISSING
    loss_coefs: Optional[List[float]] = None
    num_steps: int = MISSING
    lr: float = MISSING
    weight_decay: float = MISSING
    betas: Tuple[float] = MISSING
    lr_decay: float = MISSING


GenNetsArgs = models.generators.make_dataclass_from_args("GenNetsArgs")
args.add_to_registry("gennets")(GenNetsArgs)


@args.add_to_registry("disc")
@dataclass
class DiscArgs:
    model: str = MISSING
    optim: Optional[str] = None
    loss_funcs: Optional[List[str]] = field(default_factory=list)
    loss_coefs: Optional[List[float]] = None
    num_steps: Optional[int] = MISSING
    lr: Optional[float] = MISSING
    weight_decay: Optional[float] = MISSING
    betas: Optional[Tuple[float]] = MISSING
    lr_decay: Optional[float] = MISSING


DiscNetsArgs = models.discriminators.make_dataclass_from_args("DiscNetsArgs")
args.add_to_registry("discnets")(DiscNetsArgs)

GenOptimsArgs = optims.optims.make_dataclass_from_args("OptimsArgs")
args.add_to_registry("gen_optim")(GenOptimsArgs)

DiscOptimsArgs = optims.optims.make_dataclass_from_args("OptimsArgs")
args.add_to_registry("disc_optim")(DiscOptimsArgs)


@args.add_to_registry("metrics")
@dataclass
class MetricArgs:
    fid_cache: str = MISSING


@args.add_to_registry("checkpoint")
@dataclass
class CheckpointArgs:
    save_every: int = 100
    save_full_every: int = 200
    checkpoint_dir: str = "checkpoints/"
    checkpointing_off: bool = False
    checkpoint4inference: str = MISSING


Args = args.make_dataclass_from_classes("Args")


def load_config():
    config = OmegaConf.structured(Args)

    conf_cli = OmegaConf.from_cli()
    config.exp.config = conf_cli.exp.config
    config.exp.config_dir = conf_cli.exp.config_dir

    config_path = os.path.join(config.exp.config_dir, config.exp.config)
    conf_file = OmegaConf.load(config_path)
    config = OmegaConf.merge(config, conf_file)

    config = OmegaConf.merge(config, conf_cli)

    return config
