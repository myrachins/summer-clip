import importlib
import itertools
import typing as tp

from omegaconf import DictConfig


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
    cfg_dict: tp.Dict[str, tp.Any] = dict(cfg)
    obj_type = load_obj(cfg_dict['_target_'])
    obj_type_dict = {'_target_': type_full_name(obj_type)}
    cfg_dict.pop('_target_')

    for param_values in itertools.product(*cfg_dict.values()):
        param_to_value = dict(zip(cfg_dict.keys(), param_values))
        obj = obj_type(**param_to_value)
        obj_params = {**obj_type_dict, **param_to_value}
        yield obj, obj_params

    if len(cfg_dict) == 0:
        yield obj_type(), obj_type_dict
