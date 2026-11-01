"""PNG byte integrity and complete-radio-source Dataset contract tests."""

import hashlib
import json
import shutil
import struct
import zlib
from copy import deepcopy

import numpy as np
import pytest


def _read_png(*args, **kwargs):
    from tiangong.png import read_png

    return read_png(*args, **kwargs)


def _png(path, *, size=(4, 2), color=(255, 128, 0), mode="RGB"):
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, size, color).save(path, format="PNG")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _chunk(kind, data):
    return (struct.pack(">I", len(data)) + kind + data
            + struct.pack(">I", zlib.crc32(kind + data)))


def _raw_png(path, *, width=1, height=1, bit_depth=8, pixels=b"\x00\xff\x00\x00"):
    header = struct.pack(">IIBBBBB", width, height, bit_depth, 2, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", header)
                     + _chunk(b"IDAT", zlib.compress(pixels)) + _chunk(b"IEND", b""))


def test_read_png_rgb_preserves_non_square_shape_and_rgb_values(tmp_path):
    path = tmp_path / "input.png"
    digest = _png(path)
    image, meta = _read_png(path, expected_sha256=digest)
    assert image.shape == (3, 2, 4)
    assert image.dtype == np.float32
    assert image.flags.c_contiguous
    np.testing.assert_allclose(image[:, 0, 0], [1, 128 / 255, 0])
    assert meta == {"width": 4, "height": 2, "mode": "RGB", "sha256": digest}


def test_opaque_rgba_drops_only_alpha(tmp_path):
    path = tmp_path / "opaque.png"
    _png(path, mode="RGBA", color=(0, 64, 255, 255))
    image, meta = _read_png(path)
    np.testing.assert_allclose(image[:, 0, 0], [0, 64 / 255, 1])
    assert meta["mode"] == "RGBA"


@pytest.mark.parametrize("alpha", [0, 1, 254])
def test_any_nonopaque_rgba_pixel_is_rejected(tmp_path, alpha):
    from PIL import Image

    path = tmp_path / "transparent.png"
    source = Image.new("RGBA", (4, 2), (255, 0, 0, 255))
    source.putpixel((3, 1), (255, 0, 0, alpha))
    source.save(path)
    with pytest.raises(ValueError, match="alpha|透明|opaque"):
        _read_png(path)


@pytest.mark.parametrize("mode,color", [("L", 128), ("P", 1), ("LA", (128, 255))])
def test_unsupported_png_modes_are_rejected(tmp_path, mode, color):
    path = tmp_path / "mode.png"
    _png(path, mode=mode, color=color)
    with pytest.raises(ValueError, match="RGB|mode|模式|8-bit"):
        _read_png(path)


def test_rgb_transparency_chunk_is_rejected(tmp_path):
    from PIL import Image

    path = tmp_path / "transparent-rgb.png"
    Image.new("RGB", (2, 2), (255, 0, 0)).save(path, transparency=(255, 0, 0))
    with pytest.raises(ValueError, match="alpha|透明|opaque|transparency"):
        _read_png(path)


def test_16_bit_rgb_is_not_silently_downcast(tmp_path):
    path = tmp_path / "16-bit.png"
    _raw_png(path, bit_depth=16, pixels=b"\x00\xff\xff\x00\x00\x00\x00")
    with pytest.raises(ValueError, match="8.bit|位深|depth"):
        _read_png(path)


def test_non_png_disguised_with_png_extension_is_rejected(tmp_path):
    from PIL import Image

    path = tmp_path / "fake.png"
    Image.new("RGB", (4, 2)).save(path, format="JPEG")
    with pytest.raises(ValueError, match="PNG"):
        _read_png(path)


def test_apng_is_rejected(tmp_path):
    from PIL import Image

    path = tmp_path / "animation.png"
    Image.new("RGB", (4, 2), "red").save(
        path, save_all=True, append_images=[Image.new("RGB", (4, 2), "blue")], duration=10,
    )
    with pytest.raises(ValueError, match="APNG|动画|animation"):
        _read_png(path)


@pytest.mark.parametrize("damage", ["truncate", "crc", "iend-crc", "trailing"])
def test_corrupted_png_bytes_are_rejected(tmp_path, damage):
    path = tmp_path / "broken.png"
    _png(path)
    content = bytearray(path.read_bytes())
    if damage == "truncate":
        content = content[:-15]
    elif damage == "crc":
        content[content.index(b"IDAT") + 4] ^= 1
    elif damage == "iend-crc":
        content[-1] ^= 1
    else:
        content += b"unexpected trailing bytes"
    path.write_bytes(content)
    with pytest.raises(ValueError, match="PNG|CRC|截断|truncat"):
        _read_png(path)


def test_verified_chunks_with_invalid_deflate_pixels_fail_full_decode(tmp_path):
    path = tmp_path / "invalid-scanline.png"
    _raw_png(path, pixels=b"\xff\xff\x00\x00")  # Invalid PNG scanline filter 255.
    with pytest.raises(ValueError, match="PNG|解码|decod"):
        _read_png(path)


@pytest.mark.parametrize("damage", ["truncated-checksum", "bad-checksum"])
def test_valid_png_crcs_do_not_hide_broken_zlib_stream(tmp_path, damage):
    path = tmp_path / "bad-zlib.png"
    compressed = bytearray(zlib.compress(b"\x00\xff\x00\x00"))
    if damage == "truncated-checksum":
        compressed = compressed[:-2]
    else:
        compressed[-1] ^= 1
    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", header)
                     + _chunk(b"IDAT", compressed) + _chunk(b"IEND", b""))
    with pytest.raises(ValueError, match="PNG|zlib|解码|decod"):
        _read_png(path)


@pytest.mark.parametrize("digest", ["0" * 64, "xyz", True])
def test_wrong_or_malformed_expected_hash_is_rejected(tmp_path, digest):
    path = tmp_path / "input.png"
    _png(path)
    with pytest.raises(ValueError, match="SHA256|sha256"):
        _read_png(path, expected_sha256=digest)


def test_compressed_image_with_excessive_dimensions_rejected_before_decode(tmp_path):
    path = tmp_path / "oversized.png"
    _raw_png(path, width=100_000, height=100_000)
    with pytest.raises(ValueError, match="像素|pixel|尺寸"):
        _read_png(path)


def test_file_byte_limit_is_enforced(tmp_path):
    path = tmp_path / "large.png"
    with path.open("wb") as stream:
        stream.truncate(64 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="64|大小|byte|MiB"):
        _read_png(path)


def test_missing_png_is_explicit_file_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        _read_png(tmp_path / "missing.png")


def _dataset(*args, **kwargs):
    from tiangong.source_data import SourceDataset

    return SourceDataset(*args, **kwargs)


def _source_payload(root, *, empty=False):
    digest = _png(root / "images" / "sample.png", size=(8, 4))
    sources = [] if empty else [
        {"source_id": "annotation-10", "category_id": 0, "label": 0,
         "bbox": [0, 0, 4, 2], "segmentation": [[0, 0, 4, 0, 4, 2, 0, 2]],
         "area": 8, "original_annotation_ids": [10]},
        {"source_id": "annotation-11", "category_id": 3, "label": 3,
         "bbox": [4, 2, 8, 4], "segmentation": [], "area": 8,
         "original_annotation_ids": [11]},
    ]
    return {
        "schema_version": 1, "task": "source_detection",
        "annotation_unit": "complete_radio_source", "coordinate_system": "pixel_xyxy",
        "preprocessing": {"version": "png-rgb-unit-v1", "channels": 3,
                          "alpha_policy": "require_opaque", "resize": None},
        "categories": [{"id": index, "name": name, "label": index}
                       for index, name in enumerate(["CJ", "CS", "FRI", "FRII"])],
        "provenance": {"coco_sha256": "a" * 64, "coco_file_name": "result.json",
                       "policy": {"duplicate_policy": "error", "invalid_policy": "error",
                                  "allow_empty": empty}},
        "partition": {"status": "unassigned", "grouping_provenance": None,
                      "scientific_independence": "unverified"},
        "samples": [{"sample_id": "image-1", "file_name": "images/sample.png",
                     "sha256": digest, "width": 8, "height": 4,
                     "original_image_ids": [1], "group_id": None, "split": None,
                     "sources": sources}],
        "review": [], "changes": [],
        "summary": {"accepted_images": 1, "accepted_annotations": len(sources),
                    "review_images": 0,
                    "annotations_by_class": {"CJ": 0 if empty else 1, "CS": 0,
                                     "FRI": 0, "FRII": 0 if empty else 1}},
    }


def _write_manifest(tmp_path, payload):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_dataset_preserves_source_identity_and_normalizes_non_square_boxes(tmp_path):
    import torch

    root = tmp_path / "images-root"
    payload = _source_payload(root)
    dataset = _dataset(_write_manifest(tmp_path, payload), root)
    image, target, sample_id = dataset[0]
    assert len(dataset) == 1
    assert sample_id == "image-1"
    assert image.shape == (3, 4, 8)
    assert image.dtype == torch.float32
    assert target["labels"].dtype == torch.int64
    assert target["labels"].tolist() == [0, 3]
    assert target["boxes"].dtype == torch.float32
    assert target["boxes"].tolist() == [[0.25, 0.25, 0.5, 0.5], [0.75, 0.75, 0.5, 0.5]]
    assert target["pixel_boxes"].tolist() == [[0, 0, 4, 2], [4, 2, 8, 4]]
    assert target["source_ids"] == ["annotation-10", "annotation-11"]
    assert target["polygons"] == [[[0, 0, 4, 0, 4, 2, 0, 2]], []]
    target["polygons"][0][0][0] = 99
    assert dataset[0][1]["polygons"][0][0][0] == 0


def test_explicit_empty_source_image_returns_shape_zero_by_four(tmp_path):
    root = tmp_path / "images-root"
    dataset = _dataset(_write_manifest(tmp_path, _source_payload(root, empty=True)), root)
    _, target, _ = dataset[0]
    assert target["boxes"].shape == (0, 4)
    assert target["pixel_boxes"].shape == (0, 4)
    assert target["labels"].shape == (0,)
    assert target["source_ids"] == []
    assert target["polygons"] == []


def test_review_only_manifest_cannot_become_training_dataset(tmp_path):
    from tiangong.source_manifest import read_source_manifest

    root = tmp_path / "images-root"
    payload = _source_payload(root)
    payload["samples"] = []
    payload["provenance"]["policy"]["invalid_policy"] = "quarantine"
    payload["review"] = [{"file_name": "images/sample.png", "image_ids": [1],
                          "annotation_ids": [10, 11], "reason_codes": ["bbox_out_of_bounds"]}]
    payload["summary"] = {"accepted_images": 0, "accepted_annotations": 0,
                          "review_images": 1,
                          "annotations_by_class": {"CJ": 0, "CS": 0, "FRI": 0, "FRII": 0}}
    manifest = _write_manifest(tmp_path, payload)
    assert read_source_manifest(manifest)["samples"] == []
    with pytest.raises(ValueError, match="Dataset|没有图片"):
        _dataset(manifest, root)


def test_dataset_remains_valid_after_image_root_relocation(tmp_path):
    root = tmp_path / "old-root"
    manifest = _write_manifest(tmp_path, _source_payload(root))
    relocated = tmp_path / "new-root"
    shutil.move(root, relocated)
    image, _, sample_id = _dataset(manifest, relocated)[0]
    assert image.shape == (3, 4, 8)
    assert sample_id == "image-1"


@pytest.mark.parametrize("mutation", ["task", "extra", "bbox", "label"])
def test_dataset_rejects_wrong_task_and_malformed_manifest(tmp_path, mutation):
    root = tmp_path / "images-root"
    payload = _source_payload(root)
    if mutation == "task":
        payload["task"] = "component_detection"
    elif mutation == "extra":
        payload["silent_transform"] = "resize"
    elif mutation == "bbox":
        payload["samples"][0]["sources"][0]["bbox"] = [-1, 0, 4, 2]
    else:
        payload["samples"][0]["sources"][0]["label"] = 1
    with pytest.raises(ValueError):
        _dataset(_write_manifest(tmp_path, payload), root)


def test_dataset_checks_manifest_hash(tmp_path):
    root = tmp_path / "images-root"
    manifest = _write_manifest(tmp_path, _source_payload(root))
    with pytest.raises(ValueError, match="SHA256|sha256"):
        _dataset(manifest, root, expected_sha256="0" * 64)


@pytest.mark.parametrize("name", ["../outside.png", "/outside.png", "images/../../outside.png"])
def test_dataset_rejects_paths_outside_root(tmp_path, name):
    root = tmp_path / "images-root"
    payload = _source_payload(root)
    payload["samples"][0]["file_name"] = name
    with pytest.raises(ValueError):
        _dataset(_write_manifest(tmp_path, payload), root)


def test_dataset_rechecks_symlink_escape_on_each_access(tmp_path):
    root = tmp_path / "images-root"
    manifest = _write_manifest(tmp_path, _source_payload(root))
    dataset = _dataset(manifest, root)
    path = root / "images/sample.png"
    external = tmp_path / "outside.png"
    shutil.move(path, external)
    path.symlink_to(external)
    with pytest.raises(ValueError, match="root|根目录|路径|symlink"):
        dataset[0]
    with pytest.raises(ValueError, match="root|根目录|路径|symlink"):
        _dataset(manifest, root)


def test_dataset_rechecks_bytes_on_each_access(tmp_path):
    root = tmp_path / "images-root"
    manifest = _write_manifest(tmp_path, _source_payload(root))
    dataset = _dataset(manifest, root)
    dataset[0]
    _png(root / "images/sample.png", size=(8, 4), color=(0, 0, 255))
    with pytest.raises(ValueError, match="SHA256|sha256"):
        dataset[0]


def test_dataset_rejects_manifest_dimensions_different_from_image(tmp_path):
    root = tmp_path / "images-root"
    payload = _source_payload(root)
    payload["samples"][0]["width"] = 16
    dataset = _dataset(_write_manifest(tmp_path, payload), root)
    with pytest.raises(ValueError, match="width|尺寸|dimension"):
        dataset[0]


@pytest.mark.parametrize("split", ["unknown", "train", 1, []])
def test_unassigned_dataset_cannot_be_used_as_named_split(tmp_path, split):
    root = tmp_path / "images-root"
    with pytest.raises(ValueError, match="split|分区"):
        _dataset(_write_manifest(tmp_path, _source_payload(root)), root, split=split)


def test_dataset_selects_only_explicitly_assigned_split(tmp_path):
    from tiangong.source_manifest import assign_source_splits

    root = tmp_path / "images-root"
    payload = _source_payload(root)
    assignments = []
    template = payload["samples"][0]
    payload["samples"] = []
    for index, split in enumerate(["train", "val", "test"]):
        sample = deepcopy(template)
        sample["sample_id"] = f"image-{index + 1}"
        sample["file_name"] = f"images/sample-{index}.png"
        sample["sha256"] = _png(root / sample["file_name"], size=(8, 4), color=(index, 0, 0))
        sample["original_image_ids"] = [index + 1]
        for offset, source in enumerate(sample["sources"]):
            annotation_id = 10 + index * 2 + offset
            source["source_id"] = f"annotation-{annotation_id}"
            source["original_annotation_ids"] = [annotation_id]
        payload["samples"].append(sample)
        assignments.append({"sample_id": sample["sample_id"], "group_id": f"group-{index}", "split": split})
    payload["summary"].update(accepted_images=3, accepted_annotations=6,
                              annotations_by_class={"CJ": 3, "CS": 0, "FRI": 0, "FRII": 3})
    assigned = assign_source_splits(payload, assignments, grouping_provenance="controlled fixture groups")
    dataset = _dataset(_write_manifest(tmp_path, assigned), root, split="val")
    assert len(dataset) == 1
    assert dataset[0][2] == "image-2"


def test_bbox_only_source_accepts_omitted_segmentation(tmp_path):
    root = tmp_path / "images-root"
    payload = _source_payload(root)
    del payload["samples"][0]["sources"][1]["segmentation"]
    dataset = _dataset(_write_manifest(tmp_path, payload), root)
    assert dataset[0][1]["polygons"][1] == []


def test_source_collate_keeps_variable_image_sizes_and_empty_targets(tmp_path):
    from tiangong.source_data import source_collate_fn

    first_root = tmp_path / "first"
    first = _dataset(_write_manifest(tmp_path, _source_payload(first_root)), first_root)[0]
    second_root = tmp_path / "second"
    payload = _source_payload(second_root, empty=True)
    sample = payload["samples"][0]
    sample["width"], sample["height"] = 3, 5
    sample["sha256"] = _png(second_root / "images/sample.png", size=(3, 5))
    second = _dataset(_write_manifest(tmp_path, payload), second_root)[0]
    images, targets, ids = source_collate_fn([first, second])
    assert isinstance(images, list) and isinstance(targets, list) and isinstance(ids, list)
    assert [tuple(image.shape) for image in images] == [(3, 4, 8), (3, 5, 3)]
    assert targets[1]["boxes"].shape == (0, 4)
    assert ids == ["image-1", "image-1"]
    with pytest.raises(ValueError, match="empty|空"):
        source_collate_fn([])
