"""分量标注的坐标、负样本和数据隔离契约。"""
import copy
import hashlib
import json

import numpy as np
import pytest
import torch
from astropy.io import fits


def write_manifest(tmp_path, samples=None):
    """只在临时目录构造标注，期望框值由测试手算。"""
    if samples is None:
        samples = [{
            "sample_id": "sample_a", "path": "a.fits", "split": "train",
            "group_id": "sky_a", "components": [{
                "component_id": "component_1", "label": 0,
                "box": [0, 4, 16, 12],
            }],
        }]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({
        "schema_version": 1, "coordinate_system": "pixel_xyxy", "samples": samples,
    }), encoding="utf-8")
    return path


def test_fits_to_tensor_and_box_coordinate_contract(tmp_path):
    from detection_data import DetectionDataset, read_detection_manifest

    pixels = np.arange(256, dtype=np.float32).reshape(16, 16)
    pixels[0, 0] = np.nan
    fits.writeto(tmp_path / "a.fits", pixels)
    manifest = write_manifest(tmp_path)
    rows = read_detection_manifest(manifest, 1)
    image, target = DetectionDataset(rows, 16)[0]
    assert image.shape == (2, 16, 16)
    assert image.dtype == torch.float32
    assert torch.isfinite(image).all()
    assert image[1, 0, 0].item() == 0
    assert target["labels"].tolist() == [0]
    # [0,4,16,12] 的中心是 [8,8]、宽高是 [16,8]。
    torch.testing.assert_close(target["boxes"], torch.tensor([[0.5, 0.5, 1.0, 0.5]]))
    assert target["sample_id"] == "sample_a"
    assert target["component_ids"] == ["component_1"]
    assert target["path"] == str((tmp_path / "a.fits").resolve())


def test_empty_component_image_collates_with_positive_image(tmp_path):
    from detection_data import DetectionDataset, detection_collate, read_detection_manifest

    fits.writeto(tmp_path / "a.fits", np.zeros((16, 16), dtype=np.float32))
    first = json.loads(write_manifest(tmp_path).read_text())["samples"][0]
    second = dict(first, sample_id="sample_b", path="b.fits", components=[])
    fits.writeto(tmp_path / "b.fits", np.zeros((16, 16), dtype=np.float32))
    rows = read_detection_manifest(write_manifest(tmp_path, [first, second]), 1)
    dataset = DetectionDataset(rows, 16)
    images, targets = detection_collate([dataset[0], dataset[1]])
    assert images.shape == (2, 2, 16, 16)
    assert targets[1]["boxes"].shape == (0, 4)
    assert targets[1]["labels"].shape == (0,)
    assert targets[1]["labels"].dtype == torch.int64


@pytest.mark.parametrize("field,value", [
    ("group_id", "sky_a "), ("sample_id", " sample_a"),
    ("group_id", ""), ("sample_id", None), ("split", "validation"),
])
def test_manifest_rejects_ambiguous_ids_and_bad_split(tmp_path, field, value):
    from detection_data import read_detection_manifest

    sample = json.loads(write_manifest(tmp_path).read_text())["samples"][0]
    sample[field] = value
    with pytest.raises(ValueError):
        read_detection_manifest(write_manifest(tmp_path, [sample]), 1)


@pytest.mark.parametrize("leak", ["group", "path", "sample_id"])
def test_manifest_rejects_duplicate_or_cross_split_identity(tmp_path, leak):
    from detection_data import read_detection_manifest

    first = json.loads(write_manifest(tmp_path).read_text())["samples"][0]
    second = dict(first, sample_id="sample_b", path="b.fits", split="test", group_id="sky_b")
    if leak == "group":
        second["group_id"] = first["group_id"]
    elif leak == "path":
        second["path"] = "./a.fits"
    else:
        second["sample_id"] = first["sample_id"]
    with pytest.raises(ValueError):
        read_detection_manifest(write_manifest(tmp_path, [first, second]), 1)


@pytest.mark.parametrize("component", [
    {"component_id": "c", "label": -1, "box": [0, 1, 2, 3]},
    {"component_id": "c", "label": 1, "box": [0, 1, 2, 3]},
    {"component_id": "c", "label": True, "box": [0, 1, 2, 3]},
    {"component_id": "c", "label": 0, "box": [2, 1, 2, 3]},
    {"component_id": "c", "label": 0, "box": [-1, 1, 2, 3]},
    {"component_id": "c", "label": 0, "box": [0, 1, float("nan"), 3]},
    {"component_id": "c", "label": 0, "box": [0, 1, float("inf"), 3]},
    {"component_id": "", "label": 0, "box": [0, 1, 2, 3]},
])
def test_manifest_rejects_invalid_components(tmp_path, component):
    from detection_data import read_detection_manifest

    sample = json.loads(write_manifest(tmp_path).read_text())["samples"][0]
    sample["components"] = [component]
    with pytest.raises(ValueError):
        read_detection_manifest(write_manifest(tmp_path, [sample]), 1)


def test_duplicate_component_id_is_rejected(tmp_path):
    from detection_data import read_detection_manifest

    sample = json.loads(write_manifest(tmp_path).read_text())["samples"][0]
    sample["components"] *= 2
    with pytest.raises(ValueError, match="component_id"):
        read_detection_manifest(write_manifest(tmp_path, [sample]), 1)


@pytest.mark.parametrize("change", ["version", "coordinates", "samples", "num_classes"])
def test_manifest_requires_supported_schema(tmp_path, change):
    from detection_data import read_detection_manifest

    path = write_manifest(tmp_path)
    payload = json.loads(path.read_text())
    classes = 1
    if change == "version":
        payload["schema_version"] = 2
    elif change == "coordinates":
        payload["coordinate_system"] = "world_radec"
    elif change == "samples":
        payload["samples"] = []
    else:
        classes = 0
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        read_detection_manifest(path, classes)


def test_dataset_rejects_outside_box_and_implicit_resizing(tmp_path):
    from detection_data import DetectionDataset, read_detection_manifest

    fits.writeto(tmp_path / "a.fits", np.zeros((16, 16), dtype=np.float32))
    rows = read_detection_manifest(write_manifest(tmp_path), 1)
    with pytest.raises(ValueError, match="尺寸|size"):
        DetectionDataset(rows, 32)[0]
    broken = copy.deepcopy(rows)
    broken[0]["components"][0]["box"] = [0, 0, 17, 16]
    with pytest.raises(ValueError, match="box|框"):
        DetectionDataset(broken, 16)[0]


def test_float32_extreme_constant_keeps_zero_signal():
    from data import preprocess

    with np.errstate(over="raise", invalid="raise"):
        result = preprocess(np.full((8, 8), np.finfo(np.float32).max, dtype=np.float32))
    np.testing.assert_array_equal(result[0], np.zeros((8, 8), dtype=np.float32))
    np.testing.assert_array_equal(result[1], np.ones((8, 8), dtype=np.float32))


def test_manifest_digest_validates_the_bytes_that_are_parsed(tmp_path):
    from detection_data import read_detection_manifest

    path = write_manifest(tmp_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = read_detection_manifest(path, 1, expected_sha256=digest)
    assert rows[0]["split"] == "train"
    payload = json.loads(path.read_text())
    payload["samples"][0]["split"] = "test"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="SHA256|sha256|清单"):
        read_detection_manifest(path, 1, expected_sha256=digest)
