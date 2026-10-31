"""集合检测器的可观察契约：匹配、空图、真实梯度和小批量学习。"""

import math

import pytest
import torch

from detection_losses import (
    box_cxcywh_to_xyxy,
    box_iou,
    detection_loss,
    generalized_box_iou,
    match_predictions,
)
from detection_model import TinySetDetector


@pytest.fixture(autouse=True)
def small_cpu_workload():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _target(labels=(), boxes=()):
    return {
        "labels": torch.tensor(labels, dtype=torch.int64),
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
    }


def _outputs():
    return {
        "logits": torch.tensor([[[5.0, -5.0], [-5.0, 5.0]]]),
        "boxes": torch.tensor([[[0.25, 0.25, 0.2, 0.2], [0.75, 0.75, 0.2, 0.2]]]),
    }


def test_box_conversion_uses_continuous_edges():
    converted = box_cxcywh_to_xyxy(torch.tensor([[0.5, 0.4, 0.6, 0.2]]))
    assert torch.allclose(converted, torch.tensor([[0.2, 0.3, 0.8, 0.5]]))


def test_iou_and_giou_match_hand_calculated_areas():
    first = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
    second = torch.tensor([[1.0, 1.0, 3.0, 3.0], [3.0, 0.0, 4.0, 1.0]])
    # 第一对交集为 1，union 为 7，最小外接矩形面积为 9。
    assert torch.allclose(box_iou(first, second), torch.tensor([[1.0 / 7.0, 0.0]]))
    assert torch.allclose(
        generalized_box_iou(first, second),
        torch.tensor([[1.0 / 7.0 - 2.0 / 9.0, -3.0 / 8.0]]),
    )


def test_box_metrics_support_empty_and_zero_area_predictions():
    empty = torch.empty((0, 4))
    point = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    assert box_iou(empty, point).shape == (0, 1)
    assert generalized_box_iou(point, empty).shape == (1, 0)
    assert box_iou(point, point).item() == 0.0
    assert torch.isfinite(generalized_box_iou(point, point)).all()


@pytest.mark.parametrize("bad", [torch.ones(4), torch.tensor([[0.0, 0.0, float("nan"), 1.0]]),
                                    torch.tensor([[1.0, 0.0, 0.0, 1.0]])])
def test_box_metrics_reject_malformed_or_inverted_boxes(bad):
    with pytest.raises((ValueError, TypeError)):
        box_iou(bad, torch.tensor([[0.0, 0.0, 1.0, 1.0]]))


def test_matching_follows_boxes_when_target_order_is_swapped():
    outputs = {
        "logits": torch.tensor([[[8.0, -8.0], [8.0, -8.0]]]),
        "boxes": torch.tensor([[[0.2, 0.3, 0.2, 0.2], [0.8, 0.7, 0.2, 0.2]]]),
    }
    targets = [_target([0, 0], [[0.8, 0.7, 0.2, 0.2], [0.2, 0.3, 0.2, 0.2]])]
    predictions, truth = match_predictions(outputs, targets)[0]
    assert predictions.tolist() == [0, 1]
    assert truth.tolist() == [1, 0]
    original = detection_loss(outputs, targets)["loss"]
    reordered = [{"labels": targets[0]["labels"].flip(0), "boxes": targets[0]["boxes"].flip(0)}]
    assert torch.allclose(original, detection_loss(outputs, reordered)["loss"])


def test_matching_uses_class_probability_when_geometry_ties():
    outputs = {
        "logits": torch.tensor([[[8.0, -8.0, -8.0], [-8.0, 8.0, -8.0]]]),
        "boxes": torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]]]),
    }
    targets = [_target([1, 0], [[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]])]
    assert match_predictions(outputs, targets)[0][1].tolist() == [1, 0]


def test_no_target_has_background_loss_and_differentiable_zero_box_losses():
    outputs = {key: value.clone().requires_grad_() for key, value in _outputs().items()}
    matched = match_predictions(outputs, [_target()])
    assert matched[0][0].numel() == matched[0][1].numel() == 0
    losses = detection_loss(outputs, [_target()])
    assert losses["loss_ce"].item() > 0.0
    assert losses["loss_bbox"].item() == losses["loss_giou"].item() == 0.0
    losses["loss"].backward()
    assert outputs["logits"].grad.abs().sum() > 0
    assert torch.equal(outputs["boxes"].grad, torch.zeros_like(outputs["boxes"]))


def test_loss_has_background_weight_and_normalizes_boxes_by_target_count():
    outputs = {
        "logits": torch.tensor([[[0.0, math.log(3.0)], [0.0, math.log(3.0)]]]),
        "boxes": torch.tensor([[[0.4, 0.5, 0.2, 0.2], [0.8, 0.8, 0.1, 0.1]]]),
    }
    losses = detection_loss(outputs, [_target([0], [[0.5, 0.5, 0.2, 0.2]])])
    # 匹配框左右平移 0.1：L1=0.1，IoU=1/3，GIoU=1/3。
    expected_ce = (-math.log(0.25) - 0.1 * math.log(0.75)) / 1.1
    assert losses["loss_ce"].item() == pytest.approx(expected_ce)
    assert losses["loss_bbox"].item() == pytest.approx(0.1)
    assert losses["loss_giou"].item() == pytest.approx(2.0 / 3.0)
    assert losses["loss"].item() == pytest.approx(expected_ce + 0.5 + 4.0 / 3.0)
    duplicated = {key: value.repeat(2, 1, 1) for key, value in outputs.items()}
    targets = [_target([0], [[0.5, 0.5, 0.2, 0.2]]) for _ in range(2)]
    assert torch.allclose(losses["loss_bbox"], detection_loss(duplicated, targets)["loss_bbox"])


def test_empty_image_does_not_dilute_matched_box_loss():
    outputs = _outputs()
    target = _target([0], [[0.3, 0.25, 0.2, 0.2]])
    single = detection_loss(outputs, [target])
    batched = {key: value.repeat(2, 1, 1) for key, value in outputs.items()}
    mixed = detection_loss(batched, [target, _target()])
    assert torch.allclose(single["loss_bbox"], mixed["loss_bbox"])
    assert torch.allclose(single["loss_giou"], mixed["loss_giou"])


@pytest.mark.parametrize(
    "target",
    [
        _target([-1], [[0.5, 0.5, 0.2, 0.2]]),
        _target([1], [[0.5, 0.5, 0.2, 0.2]]),
        _target([0], [[float("nan"), 0.5, 0.2, 0.2]]),
        _target([0], [[0.05, 0.5, 0.2, 0.2]]),
        _target([0], [[0.5, 0.5, 0.0, 0.2]]),
        {"labels": torch.tensor([0.0]), "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]])},
        {"labels": torch.tensor([0]), "boxes": torch.ones(4)},
        {"labels": torch.tensor([0, 0]), "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]])},
    ],
)
def test_invalid_targets_are_rejected_before_assignment(target):
    with pytest.raises((ValueError, TypeError)):
        detection_loss(_outputs(), [target])


def test_more_targets_than_queries_is_explicit_error():
    targets = [_target([0, 0, 0], [[0.5, 0.5, 0.2, 0.2]] * 3)]
    with pytest.raises(ValueError, match="quer|查询|目标"):
        match_predictions(_outputs(), targets)


@pytest.mark.parametrize("mutation", ["nan_logits", "nan_boxes", "bad_logits_shape", "bad_boxes_shape", "batch_count"])
def test_invalid_outputs_are_rejected(mutation):
    outputs, targets = _outputs(), [_target()]
    if mutation == "nan_logits":
        outputs["logits"][0, 0, 0] = float("nan")
    elif mutation == "nan_boxes":
        outputs["boxes"][0, 0, 0] = float("nan")
    elif mutation == "bad_logits_shape":
        outputs["logits"] = outputs["logits"][0]
    elif mutation == "bad_boxes_shape":
        outputs["boxes"] = outputs["boxes"][..., :3]
    else:
        targets = []
    with pytest.raises((ValueError, TypeError)):
        detection_loss(outputs, targets)


def test_model_shapes_and_normalized_boxes_for_rectangular_images():
    model = TinySetDetector(num_classes=2, num_queries=4, hidden_dim=32)
    outputs = model(torch.randn(2, 2, 24, 40))
    assert outputs["logits"].shape == (2, 4, 3)
    assert outputs["boxes"].shape == (2, 4, 4)
    assert torch.isfinite(outputs["logits"]).all()
    assert ((outputs["boxes"] > 0) & (outputs["boxes"] < 1)).all()


@pytest.mark.parametrize("kwargs", [{"num_classes": 0}, {"num_classes": True},
                                    {"num_classes": 1, "num_queries": 0},
                                    {"num_classes": 1, "hidden_dim": 10}])
def test_invalid_architecture_is_rejected(kwargs):
    with pytest.raises((ValueError, TypeError)):
        TinySetDetector(**kwargs)


@pytest.mark.parametrize("images", [torch.ones(2, 1, 16, 16), torch.ones(2, 2, 16),
                                    torch.ones(2, 2, 16, 16, dtype=torch.int64),
                                    torch.full((1, 2, 16, 16), float("nan"))])
def test_model_rejects_invalid_images(images):
    with pytest.raises((ValueError, TypeError)):
        TinySetDetector(1)(images)


def test_loss_reaches_both_classification_and_box_heads():
    torch.manual_seed(5)
    model = TinySetDetector(1, num_queries=3, hidden_dim=32)
    outputs = model(torch.randn(2, 2, 24, 24))
    outputs["logits"].retain_grad()
    outputs["boxes"].retain_grad()
    targets = [_target([0], [[0.3, 0.4, 0.2, 0.3]]), _target()]
    detection_loss(outputs, targets)["loss"].backward()
    for key in ("logits", "boxes"):
        assert outputs[key].grad is not None
        assert torch.isfinite(outputs[key].grad).all()
        assert outputs[key].grad.abs().sum() > 0
    for head in (model.class_head, model.box_head):
        assert sum(parameter.grad.abs().sum().item() for parameter in head.parameters()) > 0


def test_model_overfits_two_distinct_synthetic_images():
    torch.manual_seed(11)
    model = TinySetDetector(1, num_queries=3, hidden_dim=32)
    images = torch.zeros(2, 2, 24, 24)
    images[0, :, 4:10, 4:10] = 1.0
    images[1, :, 14:20, 14:20] = 1.0
    targets = [_target([0], [[0.3, 0.3, 0.25, 0.25]]),
               _target([0], [[0.7, 0.7, 0.25, 0.25]])]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    initial = detection_loss(model(images), targets)["loss"].item()
    for _ in range(100):
        optimizer.zero_grad()
        loss = detection_loss(model(images), targets)["loss"]
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        final = detection_loss(outputs, targets)
        matched = match_predictions(outputs, targets)
    assert final["loss"].item() < initial * 0.25
    assert final["loss_bbox"].item() < 0.08
    for batch_index, (predictions, _) in enumerate(matched):
        assert outputs["logits"][batch_index, predictions].softmax(-1)[0, 0] > 0.9


@pytest.mark.skipif(not torch.cuda.is_available(), reason="当前环境没有 CUDA")
def test_matching_handles_cuda_targets_and_cpu_scipy_assignment():
    outputs = {key: value.cuda() for key, value in _outputs().items()}
    target = {key: value.cuda() for key, value in _target([0], [[0.25, 0.25, 0.2, 0.2]]).items()}
    matches = match_predictions(outputs, [target])
    assert matches[0][0].device.type == "cuda"
    assert matches[0][1].device.type == "cuda"
    assert torch.isfinite(detection_loss(outputs, [target])["loss"])
