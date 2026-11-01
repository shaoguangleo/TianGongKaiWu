"""分量检测数据：显式像素坐标、背景空图及完整清单隔离检查。"""
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import preprocess, read_image

PREPROCESS_VERSION = "radio-asinh-mask-v1-f64"


def _identifier(value: Any, field: str) -> str:
    """拒绝含首尾空白的标识，避免同一组通过空白差异绕过隔离。"""
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field} 必须是无首尾空白的非空字符串")
    return value


def read_detection_manifest(
    path: str | Path, num_classes: int, *, expected_sha256: str | None = None,
) -> list[dict]:
    """读取独立 JSON 标注；相对路径始终以清单目录为根。

    此函数仅检查标注和分区身份；group_id 必须由上游依据天空重叠关系
    定义，不能用字符串检查代替真实天空区域审计。
    """
    if type(num_classes) is not int or num_classes < 1:
        raise ValueError("num_classes 必须是正整数")
    manifest = Path(path).resolve()
    # 校验与解析使用同一份字节，避免文件更改后仍声称使用冻结的测试划分。
    content = manifest.read_bytes()
    if expected_sha256 is not None and hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("清单 SHA256 与训练冻结版本不一致，拒绝改变测试身份或标签")
    payload = json.loads(content.decode("utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("标注顶层必须是 JSON object")
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise ValueError("仅支持 schema_version=1")
    if payload.get("coordinate_system") != "pixel_xyxy":
        raise ValueError("coordinate_system 必须为 pixel_xyxy")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("samples 必须是非空列表")

    sample_ids: set[str] = set()
    paths: set[str] = set()
    groups: dict[str, str] = {}
    rows = []
    required = {"sample_id", "path", "split", "group_id", "components"}
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict) or not required.issubset(sample):
            raise ValueError(f"samples[{index}] 缺少必要字段 {sorted(required)}")
        sid = _identifier(sample["sample_id"], "sample_id")
        gid = _identifier(sample["group_id"], f"{sid}: group_id")
        split = sample["split"]
        if not isinstance(split, str) or split not in {"train", "val", "test"}:
            raise ValueError(f"{sid}: split 必须为 train/val/test")
        if sid in sample_ids:
            raise ValueError(f"sample_id 重复: {sid}")
        sample_ids.add(sid)
        if groups.setdefault(gid, split) != split:
            raise ValueError(f"group_id 跨分区，存在泄漏: {gid}")
        relative = _identifier(sample["path"], f"{sid}: path")
        image_path = str((manifest.parent / relative).resolve())
        if image_path in paths:
            raise ValueError(f"图像 path 重复或跨分区: {image_path}")
        paths.add(image_path)

        components = sample["components"]
        if not isinstance(components, list):
            raise ValueError(f"{sid}: components 必须为列表，负样本使用 []")
        component_ids: set[str] = set()
        parsed = []
        for component in components:
            if not isinstance(component, dict):
                raise ValueError(f"{sid}: component 必须为 JSON object")
            cid = _identifier(component.get("component_id"), f"{sid}: component_id")
            if cid in component_ids:
                raise ValueError(f"{sid}: component_id 重复: {cid}")
            component_ids.add(cid)
            label = component.get("label")
            if type(label) is not int or not 0 <= label < num_classes:
                raise ValueError(f"{sid}/{cid}: label 必须是类别范围内的整数")
            box = component.get("box")
            if (not isinstance(box, list) or len(box) != 4
                    or any(type(v) not in (int, float) for v in box)):
                raise ValueError(f"{sid}/{cid}: box 必须是四个数值的 xyxy 列表")
            try:
                x1, y1, x2, y2 = [float(v) for v in box]
            except OverflowError as exc:
                raise ValueError(f"{sid}/{cid}: box 超出数值范围") from exc
            if (not all(math.isfinite(v) for v in (x1, y1, x2, y2))
                    or not (0 <= x1 < x2 and 0 <= y1 < y2)):
                raise ValueError(f"{sid}/{cid}: box 必须有限，非负，且有正面积")
            parsed.append({"component_id": cid, "label": label, "box": [x1, y1, x2, y2]})
        rows.append({
            "sample_id": sid, "path": image_path, "split": split,
            "group_id": gid, "components": parsed,
        })
    return rows


class DetectionDataset(Dataset):
    """FITS -> 双通道张量与可变数量目标，不隐式缩放或裁剪。"""

    def __init__(self, rows: list[dict], size: int):
        if type(size) is not int or size < 8:
            raise ValueError("size 必须是 >=8 的整数")
        self.rows = rows
        self.size = size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict]:
        row = self.rows[index]
        image = read_image(row["path"])
        if image.shape != (self.size, self.size):
            raise ValueError(
                f"{row['sample_id']}: 输入尺寸应为 {(self.size, self.size)}，实际为 {image.shape}"
            )
        boxes = []
        for component in row["components"]:
            x1, y1, x2, y2 = component["box"]
            if not (0 <= x1 < x2 <= self.size and 0 <= y1 < y2 <= self.size):
                raise ValueError(f"{row['sample_id']}: box 超出图像范围或没有正面积")
            boxes.append([
                (x1 + x2) / (2 * self.size), (y1 + y2) / (2 * self.size),
                (x2 - x1) / self.size, (y2 - y1) / self.size,
            ])
        target = {
            "sample_id": row["sample_id"], "path": row["path"],
            "component_ids": [c["component_id"] for c in row["components"]],
            "labels": torch.tensor([c["label"] for c in row["components"]], dtype=torch.int64),
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        }
        # preprocess 总是返回 native-endian float32，避免 FITS big-endian 张量错误。
        return torch.from_numpy(np.ascontiguousarray(preprocess(image))), target


def detection_collate(batch: list[tuple[torch.Tensor, dict]]) -> tuple[torch.Tensor, list[dict]]:
    """图像尺寸固定，目标保留列表；空图也必须留在 batch 中。"""
    if not batch:
        raise ValueError("不能 collate 空 batch")
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)
