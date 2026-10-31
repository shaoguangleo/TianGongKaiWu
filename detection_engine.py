"""分量检测运行层：像素框、一对一评估和共享训练循环。"""
from __future__ import annotations

import math
from typing import Any, Iterable

import torch

from detection_losses import box_cxcywh_to_xyxy, box_iou, detection_loss


def _validate_threshold(value: float, name: str, *, allow_zero: bool = True) -> None:
    if not math.isfinite(value) or not (0 <= value <= 1):
        raise ValueError(f'{name}必须是[0,1]内的有限数')
    if not allow_zero and value == 0:
        raise ValueError(f'{name}必须大于0')


def _validate_size(size: int) -> None:
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError('size必须为正整数')


def postprocess_detections(
    outputs: dict[str, torch.Tensor], size: int, score_threshold: float = 0.5,
) -> list[list[dict[str, Any]]]:
    """保留前景最高且超过阈值的查询，输出裁剪后的像素 xyxy 框。

    score 来自包含背景的 softmax；不实施 NMS，以保留重复框的真实误报。
    """
    _validate_size(size)
    _validate_threshold(score_threshold, 'score_threshold')
    logits, boxes = outputs['logits'], outputs['boxes']
    if (logits.ndim != 3 or logits.shape[-1] < 2 or boxes.ndim != 3
            or boxes.shape != (*logits.shape[:2], 4)):
        raise ValueError('输出需要logits[B,Q,C+1]和boxes[B,Q,4]')
    if not torch.isfinite(logits).all() or not torch.isfinite(boxes).all():
        raise ValueError('模型输出包含非有限值')
    probabilities = logits.detach().softmax(-1).cpu()
    pixel_boxes = box_cxcywh_to_xyxy(boxes.detach().cpu()).mul(size).clamp(0, size)
    scores, labels = probabilities.max(-1)
    background = logits.shape[-1] - 1
    results: list[list[dict[str, Any]]] = []
    for image_scores, image_labels, image_boxes in zip(scores, labels, pixel_boxes):
        detections = []
        for score, label, box in zip(image_scores.tolist(), image_labels.tolist(), image_boxes.tolist()):
            if label == background or score < score_threshold:
                continue
            if box[0] >= box[2] or box[1] >= box[3]:
                continue
            detections.append(dict(label=label, score=score, box=box))
        detections.sort(key=lambda detection: detection['score'], reverse=True)
        results.append(detections)
    return results


def _target_pixel_boxes(target: dict[str, Any], size: int) -> torch.Tensor:
    boxes = target['boxes'].detach().cpu()
    return box_cxcywh_to_xyxy(boxes).mul(size)


def evaluate_detections(
    predictions: list[list[dict[str, Any]]], targets: list[dict[str, Any]],
    num_classes: int, size: int, iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """按分数降序、同类、IoU阈值逐图匹配；每个真值只能命中一次。

    这是固定阈值下的检测统计，不计算AP或mAP。空真值图上的预测均为FP。
    """
    _validate_size(size)
    _validate_threshold(iou_threshold, 'iou_threshold', allow_zero=False)
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 1:
        raise ValueError('num_classes必须为正整数')
    if len(predictions) != len(targets):
        raise ValueError('预测与真值图像数不一致')
    per_class = [dict(label=label, tp=0, fp=0, fn=0, support=0)
                 for label in range(num_classes)]
    matched_ious: list[float] = []
    for detections, target in zip(predictions, targets):
        if not isinstance(target, dict) or 'labels' not in target or 'boxes' not in target:
            raise ValueError('真值target必须包含labels和boxes')
        target_labels, target_boxes = target['labels'], target['boxes']
        if (not isinstance(target_labels, torch.Tensor) or target_labels.dtype != torch.int64
                or target_labels.ndim != 1):
            raise ValueError('真值labels必须为[N] int64 Tensor')
        if (not isinstance(target_boxes, torch.Tensor) or not target_boxes.is_floating_point()
                or target_boxes.ndim != 2 or target_boxes.shape[-1] != 4
                or len(target_labels) != len(target_boxes)):
            raise ValueError('真值boxes必须为与labels对齐的[N,4]浮点Tensor')
        if (not torch.isfinite(target_boxes).all()
                or ((target_boxes < 0) | (target_boxes > 1)).any()
                or (target_boxes[:, 2:] <= 0).any()):
            raise ValueError('真值boxes必须有限、归一化且具有正宽高')
        normalized_xyxy = box_cxcywh_to_xyxy(target_boxes)
        if ((normalized_xyxy < -1e-6) | (normalized_xyxy > 1 + 1e-6)).any():
            raise ValueError('真值boxes边界超出归一化图像范围')
        labels = target_labels.detach().cpu().tolist()
        truth_boxes = _target_pixel_boxes(target, size)
        if len(labels) != len(truth_boxes) or any(not 0 <= label < num_classes for label in labels):
            raise ValueError('真值标签或框数量非法')
        for label in labels:
            per_class[label]['support'] += 1
        matched: set[int] = set()
        for detection in sorted(detections, key=lambda item: item['score'], reverse=True):
            label = detection['label']
            if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < num_classes:
                raise ValueError('预测label超出类别范围')
            score, box = detection['score'], detection['box']
            if (not math.isfinite(score) or not 0 <= score <= 1 or len(box) != 4
                    or not all(math.isfinite(value) for value in box)
                    or box[0] >= box[2] or box[1] >= box[3]):
                raise ValueError('预测分数或框非法')
            candidates = [index for index, truth_label in enumerate(labels)
                          if truth_label == label and index not in matched]
            if candidates:
                overlaps = box_iou(torch.tensor([box], dtype=torch.float32), truth_boxes[candidates])[0]
                best_overlap, best_position = overlaps.max(0)
                if best_overlap.item() >= iou_threshold:
                    matched.add(candidates[best_position.item()])
                    per_class[label]['tp'] += 1
                    matched_ious.append(best_overlap.item())
                    continue
            per_class[label]['fp'] += 1
        for index, label in enumerate(labels):
            if index not in matched:
                per_class[label]['fn'] += 1

    def add_rates(row: dict[str, Any]) -> None:
        tp, fp, fn = row['tp'], row['fp'], row['fn']
        row['precision'] = tp / (tp + fp) if tp + fp else 0.
        row['recall'] = tp / (tp + fn) if tp + fn else 0.
        row['f1'] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.

    for row in per_class:
        add_rates(row)
    metrics = {key: sum(row[key] for row in per_class) for key in ('tp', 'fp', 'fn')}
    add_rates(metrics)
    metrics.update(
        mean_matched_iou=sum(matched_ious) / len(matched_ious) if matched_ious else 0.,
        support=[row['support'] for row in per_class], per_class=per_class,
        num_samples=len(targets), iou_threshold=iou_threshold,
        metric='score_ordered_one_to_one_iou',
        note='固定置信度和IoU阈值的一对一检测统计，不是mAP；分量框不是物理源或科学测量。',
    )
    return metrics


def run_detection_epoch(
    model: torch.nn.Module, loader: Iterable, device: torch.device,
    num_classes: int, optimizer: torch.optim.Optimizer | None = None, *,
    size: int = 64, score_threshold: float = 0.5, iou_threshold: float = 0.5,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """optimizer存在时训练；评估关闭梯度且不执行参数更新。"""
    _validate_size(size)
    _validate_threshold(score_threshold, 'score_threshold')
    _validate_threshold(iou_threshold, 'iou_threshold', allow_zero=False)
    training = optimizer is not None
    model.train(training)
    all_predictions, all_targets, records = [], [], []
    loss_sums = {key: 0. for key in ('loss', 'loss_ce', 'loss_bbox', 'loss_giou')}
    count = 0
    with torch.set_grad_enabled(training):
        for images, targets in loader:
            if len(targets) == 0 or images.shape[0] != len(targets):
                raise ValueError('batch图像与目标数量不一致或为空')
            images = images.to(device)
            device_targets = [
                {key: value.to(device) if isinstance(value, torch.Tensor) else value
                 for key, value in target.items()} for target in targets
            ]
            if training:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            if outputs['logits'].shape[-1] != num_classes + 1:
                raise ValueError('模型类别数与num_classes不一致')
            predictions = postprocess_detections(outputs, size, score_threshold)
            losses = detection_loss(outputs, device_targets)
            if any(not torch.isfinite(losses[key]).all() for key in loss_sums):
                raise ValueError('loss包含非有限值，停止运行')
            if training:
                losses['loss'].backward()
                optimizer.step()
            for key in loss_sums:
                loss_sums[key] += losses[key].detach().item() * len(targets)
            count += len(targets)
            cpu_targets = [
                {key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                 for key, value in target.items()} for target in targets
            ]
            all_predictions.extend(predictions)
            all_targets.extend(cpu_targets)
            for prediction, target in zip(predictions, cpu_targets):
                truth = [dict(component_id=component_id, label=label, box=box)
                         for component_id, label, box in zip(
                             target['component_ids'], target['labels'].tolist(),
                             _target_pixel_boxes(target, size).tolist())]
                records.append(dict(sample_id=target['sample_id'], path=str(target['path']),
                                    targets=truth, detections=prediction))
    if count == 0:
        raise ValueError('空loader，不能报告训练或评估结果')
    metrics = evaluate_detections(all_predictions, all_targets, num_classes, size, iou_threshold)
    metrics.update({key: value / count for key, value in loss_sums.items()})
    metrics['score_threshold'] = score_threshold
    return metrics, records
