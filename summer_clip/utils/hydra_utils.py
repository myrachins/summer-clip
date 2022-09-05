import importlib
import itertools
import typing as tp

import hydra
from omegaconf import DictConfig, OmegaConf


def load_obj(obj_path: str, default_obj_path: str = "") -> tp.Any:
    """Extract an object from a given path.
    Taken from: https://github.com/kedro-org/kedro/blob/e78990c6b606a27830f0d502afa0f639c0830950/kedro/utils.py#L8
    Args:
        obj_path: Path to an object to be extracted, including the object name.
        default_obj_path: Default object path.
    Returns:
        Extracted object.
    Raises:
        AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f"Object `{obj_name}` cannot be loaded from `{obj_path}`.")
    return getattr(module_obj, obj_name)


def type_full_name(type_: type) -> tp.Optional[str]:
    if type_ is None:
        return None
    module = type_.__module__
    if module is None or module == str.__module__:
        return type_.__name__
    return f'{module}.{type_.__name__}'


def instantiate_all(cfg: DictConfig) -> tp.Generator[tp.Tuple[tp.Any, tp.Dict[str, tp.Any]], None, None]:
    def create_dict_cfg(cfg: DictConfig) -> tp.Dict[str, tp.Any]:
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        return tp.cast(tp.Dict[str, tp.Any], cfg_dict)

    cfg = cfg.copy()
    cfg_dict = create_dict_cfg(cfg)
    cfg_dict.pop('_target_')

    for param_values in itertools.product(*cfg_dict.values()):
        param_to_value = dict(zip(cfg_dict.keys(), param_values))
        cfg.update(param_to_value)
        yield hydra.utils.instantiate(cfg), create_dict_cfg(cfg)
