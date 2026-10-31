"""模拟检测数据的真实 FITS、标注、确定性和文件保护测试。"""
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from astropy.io import fits


def _generate(*args, **kwargs):
    # 测试体内导入，让缺失功能表现为普通测试失败而非收集错误。
    return importlib.import_module("demo_detection").generate_demo(*args, **kwargs)


def _snapshot(directory):
    """保存所有已有字节的摘要，检查拒绝覆盖时没有改变输入。"""
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.rglob("*")
        if path.is_file()
    }


def test_each_split_has_empty_and_separated_multiple_targets(tmp_path):
    manifest = _generate(tmp_path / "demo", per_split=4, size=32, seed=7)
    assert isinstance(manifest, Path)
    document = json.loads(manifest.read_text(encoding="utf-8"))
    assert document["schema_version"] == 1
    assert document["coordinate_system"] == "pixel_xyxy"
    samples = document["samples"]
    assert len(samples) == 12
    assert len({sample["sample_id"] for sample in samples}) == 12
    assert len({sample["group_id"] for sample in samples}) == 12
    for split in ("train", "val", "test"):
        selected = [sample for sample in samples if sample["split"] == split]
        assert [len(sample["components"]) for sample in selected] == [0, 1, 2, 3]
        for sample in selected:
            relative_path = Path(sample["path"])
            assert not relative_path.is_absolute()
            assert ".." not in relative_path.parts
            image = fits.getdata(manifest.parent / relative_path)
            assert image.shape == (32, 32)
            assert image.dtype.kind == "f" and image.dtype.itemsize == 4
            assert np.isfinite(image).all()
            components = sample["components"]
            assert len({item["component_id"] for item in components}) == len(components)
            for item in components:
                assert item["label"] == 0
                x1, y1, x2, y2 = item["box"]
                assert 0 <= x1 < x2 <= 32
                assert 0 <= y1 < y2 <= 32
                # 有标注的框确实包含辐射信号，而不是仅在噪声图上附加框。
                region = image[int(y1):int(np.ceil(y2)), int(x1):int(np.ceil(x2))]
                assert region.max() > 5
            for first, left in enumerate(components):
                for right in components[first + 1:]:
                    ax1, ay1, ax2, ay2 = left["box"]
                    bx1, by1, bx2, by2 = right["box"]
                    assert ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1
            if not components:
                assert abs(float(image.mean())) < 0.2
                assert 0.8 < float(image.std()) < 1.2


def test_same_seed_reproduces_all_bytes_and_changed_seed_changes_images(tmp_path):
    first = _generate(tmp_path / "first", per_split=4, size=32, seed=0)
    second = _generate(tmp_path / "second", per_split=4, size=32, seed=0)
    third = _generate(tmp_path / "third", per_split=4, size=32, seed=1)
    assert _snapshot(first.parent) == _snapshot(second.parent)
    assert (first.parent / "train_0000.fits").read_bytes() != (
        third.parent / "train_0000.fits"
    ).read_bytes()


def test_empty_directory_is_accepted(tmp_path):
    output = tmp_path / "empty"
    output.mkdir()
    assert _generate(output, per_split=4, size=32).is_file()


@pytest.mark.parametrize("existing_name", ["manifest.json", "unrelated.bin", ".hidden"])
def test_nonempty_directory_is_rejected_without_changing_any_file(tmp_path, existing_name):
    output = tmp_path / "occupied"
    output.mkdir()
    (output / existing_name).write_bytes(b"existing content\x00\xff")
    before = _snapshot(output)
    with pytest.raises(FileExistsError):
        _generate(output, per_split=4, size=32)
    assert _snapshot(output) == before


def test_existing_output_file_is_not_overwritten(tmp_path):
    output = tmp_path / "occupied"
    output.write_bytes(b"keep this file")
    with pytest.raises(FileExistsError):
        _generate(output, per_split=4, size=32)
    assert output.read_bytes() == b"keep this file"


def test_symbolic_link_output_is_rejected(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(actual, target_is_directory=True)
    with pytest.raises(FileExistsError):
        _generate(linked, per_split=4, size=32)
    assert list(actual.iterdir()) == []


@pytest.mark.parametrize(
    "arguments",
    [
        {"per_split": 3}, {"per_split": 0}, {"per_split": 4.5}, {"per_split": True},
        {"size": 31}, {"size": 0}, {"size": 32.5}, {"size": True},
        {"seed": -1}, {"seed": 2**32}, {"seed": 1.5}, {"seed": True},
    ],
)
def test_bad_parameters_do_not_create_output(tmp_path, arguments):
    output = tmp_path / "bad"
    with pytest.raises(ValueError):
        _generate(output, **arguments)
    assert not output.exists()


def test_largest_supported_seed_is_accepted(tmp_path):
    manifest = _generate(tmp_path / "maxseed", per_split=4, size=32, seed=2**32 - 1)
    assert len(json.loads(manifest.read_text(encoding="utf-8"))["samples"]) == 12


def test_generated_manifest_is_consumed_by_real_detection_dataset(tmp_path):
    import torch
    from detection_data import DetectionDataset, detection_collate, read_detection_manifest

    manifest = _generate(tmp_path / "dataset", per_split=4, size=32)
    rows = read_detection_manifest(manifest, num_classes=1)
    dataset = DetectionDataset(rows, size=32)
    images, targets = detection_collate([dataset[index] for index in range(4)])
    assert images.shape == (4, 2, 32, 32)
    assert torch.isfinite(images).all()
    assert [target["boxes"].shape for target in targets] == [(0, 4), (1, 4), (2, 4), (3, 4)]
    for target in targets:
        assert target["labels"].dtype == torch.int64
        assert target["labels"].tolist() == [0] * len(target["boxes"])
        assert (target["boxes"] >= 0).all() and (target["boxes"] <= 1).all()


def test_cli_generates_manifest_and_rejects_overwrite(tmp_path):
    script = Path(__file__).resolve().parents[1] / "demo_detection.py"
    output = tmp_path / "cli"
    command = [
        sys.executable, str(script), "--output", str(output),
        "--per-split", "4", "--size", "32", "--seed", "9",
    ]
    first = subprocess.run(command, capture_output=True, text=True, check=False)
    assert first.returncode == 0, first.stderr
    assert (output / "manifest.json").is_file()
    before = _snapshot(output)
    second = subprocess.run(command, capture_output=True, text=True, check=False)
    assert second.returncode != 0
    assert _snapshot(output) == before


def test_cli_bad_parameter_reports_usage_without_writing(tmp_path):
    script = Path(__file__).resolve().parents[1] / "demo_detection.py"
    output = tmp_path / "bad-cli"
    result = subprocess.run(
        [sys.executable, str(script), "--output", str(output), "--seed", "-1"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "seed" in result.stderr
    assert not output.exists()
