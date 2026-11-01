"""受控模型注册表：固定架构、显式参数和完整权重校验，不动态导入模型。"""

from typing import Any

import torch
from torch import Tensor, nn

from .detection_model import TinySetDetector
from .model import MorphologyCNN


def _bounded_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    """拒绝 bool、浮点和隐式类型转换，防止配置绕过容量边界。"""
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} 必须是 [{minimum}, {maximum}] 内的整数")
    return value


def architecture_for(model: nn.Module) -> dict[str, Any]:
    """返回已注册模型的结构描述；子类可能改变计算语义，因此不接受。"""
    if type(model) is MorphologyCNN:
        return {"name": "MorphologyCNN", "version": 1, "parameters": {}}
    if type(model) is TinySetDetector:
        return {
            "name": "TinySetDetector", "version": 1,
            "parameters": {
                "num_queries": model.num_queries,
                "hidden_dim": model.hidden_dim,
            },
        }
    raise ValueError("模型架构未注册：仅支持原始 MorphologyCNN 和 TinySetDetector 类型")


def build_model(task: str, architecture: dict[str, Any], num_classes: int) -> nn.Module:
    """严格解析结构，构造 CPU float32 eval 模型，不改变默认 dtype 或 CPU RNG。"""
    if type(task) is not str or task not in {
        "morphology_classification", "component_detection",
    }:
        raise ValueError("task 必须是 morphology_classification 或 component_detection")
    if not isinstance(architecture, dict) or set(architecture) != {
        "name", "version", "parameters",
    }:
        raise ValueError("architecture 必须且只能包含 name、version、parameters")
    if type(architecture["version"]) is not int or architecture["version"] != 1:
        raise ValueError("architecture version 必须为整数 1")
    expected_name = ("MorphologyCNN" if task == "morphology_classification"
                     else "TinySetDetector")
    if type(architecture["name"]) is not str or architecture["name"] != expected_name:
        raise ValueError(f"task {task} 必须匹配架构 {expected_name}")
    parameters = architecture["parameters"]
    if not isinstance(parameters, dict):
        raise ValueError("architecture parameters 必须为 dict")
    minimum_classes = 2 if task == "morphology_classification" else 1
    _bounded_int(num_classes, "num_classes", minimum_classes, 1024)
    if task == "morphology_classification":
        if parameters:
            raise ValueError("MorphologyCNN parameters 必须为空，不接受任意 kwargs")
    else:
        if set(parameters) != {"num_queries", "hidden_dim"}:
            raise ValueError("TinySetDetector parameters 必须且只能包含 num_queries、hidden_dim")
        queries = _bounded_int(parameters["num_queries"], "num_queries", 1, 1024)
        hidden_dim = _bounded_int(parameters["hidden_dim"], "hidden_dim", 4, 512)
        if hidden_dim % 4:
            raise ValueError("hidden_dim 必须能被 4 整除")

    # 固定 CPU 构造，也覆盖调用者设置的默认 device；不触碰其他设备 RNG。
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        if task == "morphology_classification":
            model = MorphologyCNN(num_classes)
        else:
            model = TinySetDetector(num_classes, num_queries=queries, hidden_dim=hidden_dim)
    # bundle v1 固定 float32；只转换模型，不修改进程级默认 dtype。
    return model.float().eval()


def check_state_dict(model: nn.Module, state: dict[str, Tensor]) -> None:
    """先验证全部权重，再 strict 加载；成功返回 None，失败统一抛 ValueError。

    接受 CPU/GPU 上的 dense strided Tensor，包括非连续内存布局。元数据、
    稀疏、量化、嵌套 Tensor 及非有限值都不能作为这两个基线模型的权重。
    """
    if not isinstance(model, nn.Module):
        raise ValueError("model 必须是 torch.nn.Module")
    if (not isinstance(state, dict) or not state
            or any(not isinstance(key, str) or not isinstance(value, Tensor)
                   for key, value in state.items())):
        raise ValueError("state 权重必须是非空 dict[str, Tensor]")
    expected = model.state_dict()
    if set(state) != set(expected):
        missing = sorted(set(expected) - set(state))
        unexpected = sorted(set(state) - set(expected))
        raise ValueError(f"state 权重键不匹配：missing={missing}, unexpected={unexpected}")

    # 在完整检查结束之前绝不调用 load_state_dict，避免先加载合法项后才发现损坏。
    for key, value in state.items():
        reference = expected[key]
        if (value.layout != torch.strided or value.is_quantized or value.is_nested
                or value.device.type == "meta"):
            raise ValueError(f"state[{key}] 必须是有实际存储的 dense strided Tensor")
        if value.shape != reference.shape:
            raise ValueError(f"state[{key}] shape 不匹配：{tuple(value.shape)} != {tuple(reference.shape)}")
        if value.dtype != reference.dtype:
            raise ValueError(f"state[{key}] dtype 不匹配：{value.dtype} != {reference.dtype}")
        try:
            finite = bool(torch.isfinite(value).all().item())
        except (RuntimeError, TypeError, NotImplementedError) as error:
            raise ValueError(f"state[{key}] 无法验证有限值") from error
        if not finite:
            raise ValueError(f"state[{key}] 包含非有限值")
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError, ValueError, KeyError) as error:
        raise ValueError(f"state 权重无法严格加载：{error}") from error
