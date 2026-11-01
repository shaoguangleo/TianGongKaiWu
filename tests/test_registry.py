"""受控模型工厂必须拒绝不可信结构，完整恢复权重且不改变调用者 RNG。"""

import importlib
import importlib.util

import pytest
import torch

from tiangong.detection_model import TinySetDetector
from tiangong.model import MorphologyCNN


CLASSIFICATION = {
    "name": "MorphologyCNN", "version": 1, "parameters": {},
}
DETECTION = {
    "name": "TinySetDetector", "version": 1,
    "parameters": {"num_queries": 3, "hidden_dim": 8},
}


def _registry():
    # RED 阶段用明确的断言记录缺少接口，而不是让导入错误中断测试收集。
    assert importlib.util.find_spec("tiangong.registry") is not None, "缺少受控模型工厂"
    return importlib.import_module("tiangong.registry")


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("task,architecture,num_classes,expected_type", [
    ("morphology_classification", CLASSIFICATION, 2, MorphologyCNN),
    ("component_detection", DETECTION, 1, TinySetDetector),
])
def test_factory_restores_real_model_predictions_and_preserves_rng(
        task, architecture, num_classes, expected_type):
    reference = (MorphologyCNN(num_classes) if expected_type is MorphologyCNN
                 else TinySetDetector(num_classes, num_queries=3, hidden_dim=8))
    reference.eval()
    cpu_rng = torch.random.get_rng_state().clone()
    rebuilt = _registry().build_model(task, architecture, num_classes)
    assert torch.equal(torch.random.get_rng_state(), cpu_rng)
    assert type(rebuilt) is expected_type
    assert not rebuilt.training
    assert all(tensor.device.type == "cpu" for tensor in rebuilt.state_dict().values())
    assert _registry().architecture_for(reference) == architecture
    assert _registry().check_state_dict(rebuilt, reference.state_dict()) is None
    inputs = torch.linspace(-1.0, 1.0, 2 * 16 * 16).reshape(1, 2, 16, 16)
    with torch.inference_mode():
        actual, expected = rebuilt(inputs), reference(inputs)
    if expected_type is MorphologyCNN:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    else:
        for name in ("logits", "boxes"):
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


@pytest.mark.parametrize("base,kwargs", [
    (MorphologyCNN, {"num_classes": 2}),
    (TinySetDetector, {"num_classes": 1, "num_queries": 3, "hidden_dim": 8}),
])
def test_architecture_rejects_subclasses_even_when_base_weights_match(base, kwargs):
    class UnregisteredModel(base):
        pass

    with pytest.raises(ValueError, match="模型|model|架构|architecture"):
        _registry().architecture_for(UnregisteredModel(**kwargs))


@pytest.mark.parametrize("model", [None, object(), torch.nn.Linear(2, 2), "MorphologyCNN"])
def test_architecture_rejects_arbitrary_objects(model):
    with pytest.raises(ValueError):
        _registry().architecture_for(model)


@pytest.mark.parametrize("architecture", [
    None, [], "MorphologyCNN", {},
    {"name": "os.system", "version": 1, "parameters": {}},
    {"name": "torch.nn.Linear", "version": 1, "parameters": {}},
    {"name": "MorphologyCNN", "version": True, "parameters": {}},
    {"name": "MorphologyCNN", "version": 1.0, "parameters": {}},
    {"name": "MorphologyCNN", "version": 0, "parameters": {}},
    {"name": "MorphologyCNN", "version": 2, "parameters": {}},
    {"name": "MorphologyCNN", "parameters": {}},
    {"name": "MorphologyCNN", "version": 1, "parameters": {}, "module": "os"},
    {"name": "MorphologyCNN", "version": 1, "parameters": {"num_classes": 2}},
    {"name": "MorphologyCNN", "version": 1, "parameters": []},
    {"name": [], "version": 1, "parameters": {}},
])
def test_factory_rejects_unknown_architectures_and_untrusted_kwargs(architecture):
    before = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError):
        _registry().build_model("morphology_classification", architecture, 2)
    assert torch.equal(torch.random.get_rng_state(), before)


@pytest.mark.parametrize("task,architecture", [
    ("component_detection", CLASSIFICATION),
    ("morphology_classification", DETECTION),
    ("arbitrary", CLASSIFICATION),
    (None, CLASSIFICATION),
    ([], CLASSIFICATION),
])
def test_factory_rejects_task_architecture_mismatch(task, architecture):
    with pytest.raises(ValueError, match="task|任务|架构|architecture"):
        _registry().build_model(task, architecture, 2)


@pytest.mark.parametrize("task,architecture,bad_classes", [
    ("morphology_classification", CLASSIFICATION, 1),
    ("morphology_classification", CLASSIFICATION, 0),
    ("component_detection", DETECTION, 0),
    ("component_detection", DETECTION, -1),
    ("morphology_classification", CLASSIFICATION, 1025),
    ("component_detection", DETECTION, 1025),
    ("morphology_classification", CLASSIFICATION, True),
    ("component_detection", DETECTION, True),
    ("morphology_classification", CLASSIFICATION, 2.0),
    ("component_detection", DETECTION, "2"),
])
def test_factory_limits_class_count_without_bool_coercion(task, architecture, bad_classes):
    with pytest.raises(ValueError, match="num_classes"):
        _registry().build_model(task, architecture, bad_classes)


@pytest.mark.parametrize("parameters", [
    {}, {"num_queries": 3}, {"hidden_dim": 8},
    {"num_queries": 3, "hidden_dim": 8, "dropout": 0},
    {"num_queries": True, "hidden_dim": 8},
    {"num_queries": 0, "hidden_dim": 8},
    {"num_queries": 1025, "hidden_dim": 8},
    {"num_queries": 3.0, "hidden_dim": 8},
    {"num_queries": 3, "hidden_dim": True},
    {"num_queries": 3, "hidden_dim": 0},
    {"num_queries": 3, "hidden_dim": 3},
    {"num_queries": 3, "hidden_dim": 6},
    {"num_queries": 3, "hidden_dim": 516},
    {"num_queries": 3, "hidden_dim": 8.0},
])
def test_factory_validates_detection_capacity_before_construction(parameters):
    with pytest.raises(ValueError):
        _registry().build_model("component_detection", {
            "name": "TinySetDetector", "version": 1, "parameters": parameters,
        }, 1)


@pytest.mark.parametrize("task,architecture", [
    ("morphology_classification", CLASSIFICATION),
    ("component_detection", {
        "name": "TinySetDetector", "version": 1,
        "parameters": {"num_queries": 1024, "hidden_dim": 4},
    }),
    ("component_detection", {
        "name": "TinySetDetector", "version": 1,
        "parameters": {"num_queries": 1, "hidden_dim": 512},
    }),
])
def test_factory_accepts_documented_capacity_boundaries(task, architecture):
    model = _registry().build_model(task, architecture, 1024)
    assert _registry().architecture_for(model) == architecture


def test_factory_forces_cpu_even_inside_another_default_device_context():
    before = torch.random.get_rng_state().clone()
    with torch.device("meta"):
        model = _registry().build_model("morphology_classification", CLASSIFICATION, 2)
    assert torch.equal(torch.random.get_rng_state(), before)
    assert all(value.device.type == "cpu" for value in model.state_dict().values())


@pytest.mark.parametrize("task,architecture,num_classes", [
    ("morphology_classification", CLASSIFICATION, 2),
    ("component_detection", DETECTION, 1),
])
def test_factory_forces_float32_without_changing_default_dtype_or_rng(
        task, architecture, num_classes):
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        before = torch.random.get_rng_state().clone()
        model = _registry().build_model(task, architecture, num_classes)
        assert torch.get_default_dtype() is torch.float64
        assert torch.equal(torch.random.get_rng_state(), before)
        assert all(value.dtype is torch.float32 for value in model.state_dict().values())
    finally:
        torch.set_default_dtype(previous_dtype)


@pytest.mark.parametrize("invalid_state", [None, [], {}, {1: torch.ones(1)}, {"weight": 1}])
def test_weight_loader_rejects_empty_or_non_tensor_mappings(invalid_state):
    with pytest.raises(ValueError, match="state|权重|键|Tensor"):
        _registry().check_state_dict(MorphologyCNN(2), invalid_state)


@pytest.mark.parametrize("damage", [
    "missing_key", "extra_key", "shape", "dtype", "nan", "inf", "sparse", "meta", "quantized",
])
def test_weight_loader_validates_every_tensor_before_mutating_model(damage):
    model = MorphologyCNN(2)
    previous = {key: value.clone() for key, value in model.state_dict().items()}
    state = {key: torch.full_like(value, 0.25) for key, value in previous.items()}
    # 损坏最后一项；若实现一边检查一边复制，前面的合法权重会被错误写入。
    key = list(state)[-1]
    if damage == "missing_key":
        state.pop(key)
    elif damage == "extra_key":
        state["unregistered.weight"] = torch.ones(1)
    elif damage == "shape":
        state[key] = state[key].reshape(1, -1)
    elif damage == "dtype":
        state[key] = state[key].double()
    elif damage in {"nan", "inf"}:
        state[key][0] = float(damage)
    elif damage == "sparse":
        state[key] = state[key].to_sparse()
    elif damage == "meta":
        state[key] = torch.empty_like(state[key], device="meta")
    else:
        state[key] = torch.quantize_per_tensor(state[key], 0.1, 0, torch.qint8)
    with pytest.raises(ValueError):
        _registry().check_state_dict(model, state)
    for name, value in model.state_dict().items():
        assert torch.equal(value, previous[name]), name


def test_weight_loader_accepts_dense_noncontiguous_values():
    model = MorphologyCNN(2)
    state = {key: value.clone() for key, value in model.state_dict().items()}
    key = "head.2.weight"
    replacement = torch.arange(state[key].numel(), dtype=state[key].dtype).reshape(
        state[key].shape[1], state[key].shape[0]).t()
    assert not replacement.is_contiguous()
    state[key] = replacement
    _registry().check_state_dict(model, state)
    assert torch.equal(model.state_dict()[key], replacement)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="当前环境无 CUDA")
def test_weight_loader_accepts_gpu_state_for_cpu_model():
    reference = TinySetDetector(1, num_queries=3, hidden_dim=8).eval()
    state = {key: value.to("cuda") for key, value in reference.state_dict().items()}
    restored = _registry().build_model("component_detection", DETECTION, 1)
    _registry().check_state_dict(restored, state)
    for key, value in reference.state_dict().items():
        assert torch.equal(restored.state_dict()[key], value)
