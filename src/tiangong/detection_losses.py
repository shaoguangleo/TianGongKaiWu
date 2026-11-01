"""一对一集合匹配与 CE/L1/GIoU 损失，兼容含空图的可变目标数。"""

from typing import Dict, List, Mapping, Sequence, Tuple

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from torch.nn import functional as F


def _check_box_tensor(boxes: Tensor, name: str, *, matrix: bool = True) -> None:
    if not isinstance(boxes, Tensor) or not boxes.is_floating_point():
        raise TypeError(f"{name} 必须是浮点 Tensor")
    if boxes.ndim < 1 or boxes.shape[-1] != 4 or (matrix and boxes.ndim != 2):
        raise ValueError(f"{name} 必须具有 {'[N, 4]' if matrix else '[..., 4]'} 形状")
    if not torch.isfinite(boxes).all():
        raise ValueError(f"{name} 包含非有限值")


def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    """将连续中心/宽高坐标转换为连续边界；不裁剪图像外的预测。"""
    _check_box_tensor(boxes, "boxes", matrix=False)
    if (boxes[..., 2:] < 0).any():
        raise ValueError("框宽高不能为负数")
    centers, sizes = boxes[..., :2], boxes[..., 2:]
    return torch.cat((centers - sizes / 2, centers + sizes / 2), dim=-1)


def _pairwise_intersection_union(boxes1: Tensor, boxes2: Tensor) -> Tuple[Tensor, Tensor]:
    _check_box_tensor(boxes1, "boxes1")
    _check_box_tensor(boxes2, "boxes2")
    if boxes1.device != boxes2.device:
        raise ValueError("两个框集合必须位于相同 device")
    if (boxes1[:, 2:] < boxes1[:, :2]).any() or (boxes2[:, 2:] < boxes2[:, :2]).any():
        raise ValueError("xyxy 框边界倒置")
    areas1 = (boxes1[:, 2:] - boxes1[:, :2]).prod(dim=-1)
    areas2 = (boxes2[:, 2:] - boxes2[:, :2]).prod(dim=-1)
    top_left = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(dim=-1)
    union = areas1[:, None] + areas2[None, :] - intersection
    return intersection, union


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """返回连续 xyxy 框的 ``[N, M]`` IoU 矩阵，退化框的 IoU 为零。"""
    intersection, union = _pairwise_intersection_union(boxes1, boxes2)
    return intersection / union.clamp_min(torch.finfo(union.dtype).tiny)


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """返回 ``[N, M]`` GIoU 矩阵，支持空集合和零面积预测。"""
    intersection, union = _pairwise_intersection_union(boxes1, boxes2)
    epsilon = torch.finfo(union.dtype).tiny
    iou = intersection / union.clamp_min(epsilon)
    top_left = torch.minimum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.maximum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    enclosing_area = (bottom_right - top_left).clamp(min=0).prod(dim=-1)
    return iou - (enclosing_area - union) / enclosing_area.clamp_min(epsilon)


def _validated_inputs(
    outputs: Mapping[str, Tensor], targets: Sequence[Mapping[str, Tensor]],
) -> Tuple[Tensor, Tensor, List[Dict[str, Tensor]]]:
    """在进入 SciPy 或交叉熵前拒绝坏数据，目标拷贝到预测 device。"""
    if not isinstance(outputs, Mapping) or "logits" not in outputs or "boxes" not in outputs:
        raise ValueError("outputs 必须包含 logits 和 boxes")
    logits, boxes = outputs["logits"], outputs["boxes"]
    if not isinstance(logits, Tensor) or not logits.is_floating_point():
        raise TypeError("logits 必须是浮点 Tensor")
    if logits.ndim != 3 or min(logits.shape) <= 0 or logits.shape[-1] < 2:
        raise ValueError("logits 必须为非空 [B, Q, C+1]，至少一个前景类")
    _check_box_tensor(boxes, "outputs boxes", matrix=False)
    if boxes.ndim != 3 or boxes.shape != (*logits.shape[:2], 4):
        raise ValueError("outputs boxes 必须为 [B, Q, 4] 并与 logits 对齐")
    if boxes.device != logits.device:
        raise ValueError("logits 与 boxes 必须位于相同 device")
    if not torch.isfinite(logits).all():
        raise ValueError("logits 包含非有限值")
    if ((boxes < 0) | (boxes > 1)).any():
        raise ValueError("预测框必须是归一化 cxcywh")
    if not isinstance(targets, Sequence) or len(targets) != logits.shape[0]:
        raise ValueError("targets 数量必须等于 batch 大小")

    validated = []
    for target in targets:
        if not isinstance(target, Mapping) or "labels" not in target or "boxes" not in target:
            raise ValueError("每个 target 必须包含 labels 和 boxes")
        labels, target_boxes = target["labels"], target["boxes"]
        if not isinstance(labels, Tensor) or labels.dtype != torch.int64:
            raise TypeError("目标 labels 必须是 int64 Tensor")
        if labels.ndim != 1:
            raise ValueError("目标 labels 必须为 [N]")
        _check_box_tensor(target_boxes, "target boxes")
        if len(labels) != len(target_boxes):
            raise ValueError("目标 labels 数量与 boxes 数量不一致")
        if len(labels) > logits.shape[1]:
            raise ValueError("目标数超过查询数 num_queries，无法建立一对一匹配")
        if ((labels < 0) | (labels >= logits.shape[-1] - 1)).any():
            raise ValueError("目标 labels 必须是有效前景类索引，不能使用背景类")
        if ((target_boxes < 0) | (target_boxes > 1)).any() or (target_boxes[:, 2:] <= 0).any():
            raise ValueError("目标框必须归一化且具有正宽高")
        target_xyxy = box_cxcywh_to_xyxy(target_boxes)
        # 仅容忍归一化及 float32 转换引入的舍入误差。
        if ((target_xyxy < -1e-6) | (target_xyxy > 1 + 1e-6)).any():
            raise ValueError("目标框边界超出归一化图像范围")
        validated.append({
            "labels": labels.to(device=logits.device),
            "boxes": target_boxes.to(device=boxes.device, dtype=boxes.dtype),
        })
    return logits, boxes, validated


@torch.no_grad()
def _match_validated(logits: Tensor, boxes: Tensor, targets: List[Dict[str, Tensor]]) -> List[Tuple[Tensor, Tensor]]:
    assignments = []
    for image_logits, image_boxes, target in zip(logits, boxes, targets):
        if target["labels"].numel() == 0:
            empty = torch.empty(0, dtype=torch.int64, device=logits.device)
            assignments.append((empty, empty.clone()))
            continue
        classification_cost = -image_logits.softmax(dim=-1)[:, target["labels"]]
        bbox_cost = torch.cdist(image_boxes, target["boxes"], p=1)
        giou_cost = 1 - generalized_box_iou(
            box_cxcywh_to_xyxy(image_boxes), box_cxcywh_to_xyxy(target["boxes"]),
        )
        cost = classification_cost + 5 * bbox_cost + 2 * giou_cost
        # SciPy 在 CPU 求离散分配；float64 转换也兼容不能直接 .numpy() 的 dtype。
        predictions, truth = linear_sum_assignment(cost.to(device="cpu", dtype=torch.float64).numpy())
        assignments.append((
            torch.as_tensor(predictions, dtype=torch.int64, device=logits.device),
            torch.as_tensor(truth, dtype=torch.int64, device=logits.device),
        ))
    return assignments


def match_predictions(
    outputs: Mapping[str, Tensor], targets: Sequence[Mapping[str, Tensor]],
) -> List[Tuple[Tensor, Tensor]]:
    """按 ``-P(class)+5*L1+2*(1-GIoU)`` 建立逐图一对一匈牙利匹配。"""
    logits, boxes, validated = _validated_inputs(outputs, targets)
    return _match_validated(logits, boxes, validated)


def detection_loss(
    outputs: Mapping[str, Tensor], targets: Sequence[Mapping[str, Tensor]],
) -> Dict[str, Tensor]:
    """返回总损失及未加权分项；未匹配查询以权重 0.1 学习背景。"""
    logits, boxes, validated = _validated_inputs(outputs, targets)
    matches = _match_validated(logits, boxes, validated)
    background_class = logits.shape[-1] - 1
    class_targets = torch.full(logits.shape[:2], background_class, dtype=torch.int64, device=logits.device)
    matched_predictions, matched_targets = [], []
    for batch_index, ((predictions, truth), target) in enumerate(zip(matches, validated)):
        class_targets[batch_index, predictions] = target["labels"][truth]
        if predictions.numel():
            matched_predictions.append(boxes[batch_index, predictions])
            matched_targets.append(target["boxes"][truth])
    class_weights = logits.new_ones(logits.shape[-1])
    class_weights[-1] = 0.1
    loss_ce = F.cross_entropy(logits.transpose(1, 2), class_targets, weight=class_weights)
    if matched_predictions:
        predicted = torch.cat(matched_predictions)
        expected = torch.cat(matched_targets)
        normalizer = sum(len(target["labels"]) for target in validated)
        loss_bbox = F.l1_loss(predicted, expected, reduction="sum") / normalizer
        pairwise_giou = generalized_box_iou(box_cxcywh_to_xyxy(predicted), box_cxcywh_to_xyxy(expected))
        loss_giou = (1 - pairwise_giou.diagonal()).sum() / normalizer
    else:
        # 保留与框头的计算图，纯负样本 batch 仍能统一执行 backward。
        loss_bbox = boxes.sum() * 0.0
        loss_giou = boxes.sum() * 0.0
    return {
        "loss": loss_ce + 5 * loss_bbox + 2 * loss_giou,
        "loss_ce": loss_ce,
        "loss_bbox": loss_bbox,
        "loss_giou": loss_giou,
    }
