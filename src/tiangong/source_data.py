"""PNG Dataset for complete radio sources, with unchanged pixel geometry."""

from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .png import read_png
from .source_manifest import read_source_manifest


class SourceDataset(Dataset):
    """Read a strict source manifest from an explicitly relocatable image root.

    With split=None, unassigned manifests can be inspected. Selecting a training
    split requires a manifest with an explicitly assigned partition. Source
    polygons and pixel_boxes retain their original pixel coordinates.
    """

    def __init__(
        self, manifest_path: str | Path, image_root: str | Path, *,
        split: str | None = None, expected_sha256: str | None = None,
    ):
        if split is not None and (not isinstance(split, str) or split not in {"train", "val", "test"}):
            raise ValueError("split 必须为 train/val/test 或 None")
        payload = read_source_manifest(manifest_path, expected_sha256=expected_sha256)
        if split is not None and payload["partition"]["status"] != "assigned":
            raise ValueError("清单尚未分区，不能选择 split")
        self.image_root = Path(image_root).resolve(strict=True)
        if not self.image_root.is_dir():
            raise ValueError("image_root 必须是图片根目录")
        self._samples = [sample for sample in payload["samples"]
                         if split is None or sample["split"] == split]
        if not self._samples:
            raise ValueError("选中的 source Dataset 没有图片")
        for sample in self._samples:
            self._image_path(sample["file_name"])

    def _image_path(self, file_name: str) -> Path:
        """Resolve again at access time so a changed symlink cannot escape root."""
        try:
            path = (self.image_root / file_name).resolve(strict=True)
        except RuntimeError as exc:
            raise ValueError("PNG 路径含 symlink 循环") from exc
        if not path.is_relative_to(self.image_root):
            raise ValueError("PNG 路径或 symlink 越出 image_root 根目录")
        if not path.is_file():
            raise ValueError("PNG 路径必须指向普通文件")
        return path

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict, str]:
        sample = self._samples[index]
        pixels, metadata = read_png(
            self._image_path(sample["file_name"]), expected_sha256=sample["sha256"],
        )
        width, height = sample["width"], sample["height"]
        if metadata["width"] != width or metadata["height"] != height:
            raise ValueError(f"{sample['sample_id']}: PNG 尺寸与清单 width/height 不一致")
        boxes = []
        for source in sample["sources"]:
            x1, y1, x2, y2 = source["bbox"]
            boxes.append([(x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                          (x2 - x1) / width, (y2 - y1) / height])
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor([source["label"] for source in sample["sources"]], dtype=torch.int64),
            "pixel_boxes": torch.tensor(
                [source["bbox"] for source in sample["sources"]], dtype=torch.float32,
            ).reshape(-1, 4),
            "polygons": [deepcopy(source["segmentation"]) for source in sample["sources"]],
            "source_ids": [source["source_id"] for source in sample["sources"]],
        }
        return torch.from_numpy(pixels), target, sample["sample_id"]


def source_collate_fn(
    batch: list[tuple[torch.Tensor, dict, str]],
) -> tuple[list[torch.Tensor], list[dict], list[str]]:
    """Keep variable-sized images and variable-sized targets as separate lists."""
    if not batch:
        raise ValueError("不能 collate 空 batch")
    images, targets, sample_ids = zip(*batch)
    return list(images), list(targets), list(sample_ids)
