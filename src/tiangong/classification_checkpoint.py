"""分类checkpoint契约；恢复时不执行任意pickle对象或推断旧预处理版本。"""
from __future__ import annotations

import io
from pathlib import Path
import pickle
from typing import Any

import torch

from .registry import build_model, check_state_dict


CHECKPOINT_SCHEMA_VERSION = 1
TASK_NAME = 'morphology_classification'
PREPROCESS_VERSION = 'radio-asinh-mask-v1-f64'


def _positive_int(value: Any, name: str, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name}必须为>={minimum}的整数')


def _validate_classes(classes: Any) -> list[str]:
    if (not isinstance(classes, list) or not 2 <= len(classes) <= 1024
            or any(not isinstance(name, str) or not name.strip() or name != name.strip()
                   for name in classes)
            or len(set(classes)) != len(classes)):
        raise ValueError('classes需要2至1024个唯一、非空且无首尾空白的类别名')
    return classes


def _validate_size(size: Any, name: str = 'size') -> None:
    _positive_int(size, name, 8)
    if size > 4096:
        raise ValueError(f'{name}不能超过4096')


def _validate_checkpoint(state: Any, *, allow_legacy: bool = False) -> dict[str, Any]:
    """验证原始字典，不增补或重写来源无法证明的metadata。"""
    if type(allow_legacy) is not bool:
        raise ValueError('allow_legacy必须为布尔值')
    if not isinstance(state, dict):
        raise ValueError('checkpoint必须为字典')
    if 'schema_version' not in state:
        if not allow_legacy:
            raise ValueError('旧legacy checkpoint需要显式确认预处理版本后再加载')
        if {'task', 'preprocess_version', 'architecture'} & state.keys():
            raise ValueError('缺少schema_version，不能将不完整新格式降级为legacy')
    else:
        if (type(state['schema_version']) is not int
                or state['schema_version'] != CHECKPOINT_SCHEMA_VERSION):
            raise ValueError('schema_version不兼容')
        if state.get('task') != TASK_NAME:
            raise ValueError('task不兼容')
        if state.get('preprocess_version') != PREPROCESS_VERSION:
            raise ValueError('preprocess_version不兼容')
        architecture = state.get('architecture')
        if (not isinstance(architecture, dict)
                or set(architecture) != {'name', 'size'}
                or architecture.get('name') != 'MorphologyCNN'):
            raise ValueError('architecture不兼容')
        _validate_size(architecture.get('size'), 'architecture.size')
        if architecture['size'] != state.get('size'):
            raise ValueError('size与architecture.size不一致')

    classes = _validate_classes(state.get('classes'))
    _validate_size(state.get('size'))
    _positive_int(state.get('epoch'), 'epoch')
    seed = state.get('seed')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed必须为[0,2**32)内的整数')
    digest = state.get('manifest_sha256')
    if (not isinstance(digest, str) or len(digest) != 64
            or any(character not in '0123456789abcdef' for character in digest)):
        raise ValueError('manifest_sha256必须为64位小写十六进制SHA256')
    weights = state.get('model')
    if (not isinstance(weights, dict) or not weights
            or any(not isinstance(key, str) or not isinstance(value, torch.Tensor)
                   or value.layout != torch.strided or value.device.type != 'cpu'
                   or not torch.isfinite(value).all()
                   for key, value in weights.items())):
        raise ValueError('model必须包含CPU稠密有限tensor权重')

    # strict验证完整键集合与张量shape，防止将其他模型误当成本项目旧格式。
    # 验证不消耗调用者的训练随机数状态，也不接受隐式dtype转换。
    model = build_model(TASK_NAME, dict(name='MorphologyCNN', version=1, parameters={}), len(classes))
    check_state_dict(model, weights)
    return state


def load_classification_checkpoint(
    path: str | Path | io.BytesIO, *, allow_legacy: bool = False,
) -> dict[str, Any]:
    """安全恢复分类权重；legacy需显式授权，返回旧字段且不伪造版本信息。

    仅使用weights_only=True。损坏、未知schema或不匹配权重统一报ValueError；
    不回退到完整pickle，也不将不兼容现代文件当作旧格式接受。
    BytesIO用于调用者已冻结并计算摘要的字节快照；不接受任意file-like对象。
    """
    try:
        source = path if isinstance(path, io.BytesIO) else Path(path)
        state = torch.load(source, map_location='cpu', weights_only=True)
        return _validate_checkpoint(state, allow_legacy=allow_legacy)
    except (OSError, RuntimeError, ValueError, TypeError, KeyError, EOFError,
            IndexError, pickle.UnpicklingError) as error:
        raise ValueError(f'checkpoint无法安全恢复或不兼容：{error}') from error
