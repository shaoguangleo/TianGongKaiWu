"""Source COCO import: real PNGs, strict identity, and whole-image quarantine."""

import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import zlib

import pytest


def _png(path, *, width=10, height=8):
    """Create an actual RGB PNG without relying on the decoder under test."""
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(
            ">I", zlib.crc32(data, zlib.crc32(kind)),
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress((b"\x00" + b"\xff\x00\x00" * width) * height))
        + chunk(b"IEND", b"")
    )


def _annotation(identifier=20, image_id=7, **updates):
    item = {
        "id": identifier, "image_id": image_id, "category_id": 0,
        "bbox": [1, 2, 3, 4], "segmentation": [[1, 2, 4, 2, 4, 6, 1, 6]],
        "area": 12, "iscrowd": 0, "ignore": 0,
    }
    item.update(updates)
    return item


def _payload():
    return {
        "images": [{"id": 7, "file_name": "images/source.png", "width": 10, "height": 8}],
        "annotations": [_annotation()],
        "categories": [{"id": i, "name": name} for i, name in enumerate(("CJ", "CS", "FRI", "FRII"))],
    }


def _import(tmp_path, payload=None, **policies):
    # Import inside the test so missing implementation is a visible RED failure.
    from tiangong.coco_import import import_source_coco

    root = tmp_path / "data"
    _png(root / "images/source.png")
    path = tmp_path / "result.json"
    path.write_text(json.dumps(_payload() if payload is None else payload), encoding="utf-8")
    return import_source_coco(path, root, **policies)


def test_import_preserves_category_zero_polygon_and_source_provenance(tmp_path):
    manifest = _import(tmp_path)
    assert manifest["task"] == "source_detection"
    assert manifest["annotation_unit"] == "complete_radio_source"
    sample = manifest["samples"][0]
    assert sample["sample_id"] == "image-7"
    assert sample["file_name"] == "images/source.png"
    assert sample["original_image_ids"] == [7]
    assert (sample["width"], sample["height"]) == (10, 8)
    assert sample["group_id"] is None and sample["split"] is None
    assert sample["sha256"] == hashlib.sha256((tmp_path / "data/images/source.png").read_bytes()).hexdigest()
    assert sample["sources"] == [{
        "source_id": "annotation-20", "category_id": 0, "label": 0,
        "bbox": [1, 2, 4, 6], "segmentation": [[1, 2, 4, 2, 4, 6, 1, 6]],
        "area": 12, "original_annotation_ids": [20],
    }]
    assert manifest["provenance"]["coco_sha256"] == hashlib.sha256((tmp_path / "result.json").read_bytes()).hexdigest()
    assert manifest["summary"] == {
        "accepted_images": 1, "accepted_annotations": 1, "review_images": 0,
        "annotations_by_class": {"CJ": 1, "CS": 0, "FRI": 0, "FRII": 0},
    }
    assert str(tmp_path) not in json.dumps(manifest)


def test_multiple_classes_on_one_image_are_not_collapsed(tmp_path):
    data = _payload()
    data["annotations"].append(_annotation(21, category_id=3))
    manifest = _import(tmp_path, data)
    assert [source["label"] for source in manifest["samples"][0]["sources"]] == [0, 3]
    assert manifest["summary"]["accepted_annotations"] == 2


@pytest.mark.parametrize("kind", ["image_id", "annotation_id", "image_ref", "category_ref", "boolean_id", "missing_id", "nonobject_image"])
def test_global_identity_errors_reject_even_under_quarantine(tmp_path, kind):
    data = _payload()
    if kind == "image_id":
        data["images"].append(copy.deepcopy(data["images"][0]))
    elif kind == "annotation_id":
        data["annotations"].append(copy.deepcopy(data["annotations"][0]))
    elif kind == "image_ref":
        data["annotations"][0]["image_id"] = 999
    elif kind == "category_ref":
        data["annotations"][0]["category_id"] = 99
    elif kind == "boolean_id":
        data["annotations"][0]["id"] = True
    elif kind == "missing_id":
        del data["images"][0]["id"]
    else:
        data["images"][0] = "broken"
    with pytest.raises(ValueError):
        _import(tmp_path, data, invalid_policy="quarantine", duplicate_policy="merge_identical")


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong_name", "wrong_id", "boolean_id"])
def test_only_confirmed_four_category_mapping_is_accepted(tmp_path, mutation):
    data = _payload()
    if mutation == "missing":
        data["categories"].pop()
    elif mutation == "duplicate":
        data["categories"].append(copy.deepcopy(data["categories"][0]))
    elif mutation == "wrong_name":
        data["categories"][2]["name"] = "Seyfert I"
    elif mutation == "wrong_id":
        data["categories"][0]["id"] = 8
    else:
        data["categories"][0]["id"] = False
    with pytest.raises(ValueError):
        _import(tmp_path, data, invalid_policy="quarantine")


@pytest.mark.parametrize("file_name", ["../source.png", "/tmp/source.png", "images/../source.png", "./source.png", "images//source.png", "C:/source.png", "images\\source.png", "", "images/\x00source.png"])
def test_unsafe_paths_are_global_errors(tmp_path, file_name):
    data = _payload()
    data["images"][0]["file_name"] = file_name
    with pytest.raises(ValueError):
        _import(tmp_path, data, invalid_policy="quarantine")


def test_symlink_escape_rejected_even_under_quarantine(tmp_path):
    root = tmp_path / "data"
    _png(tmp_path / "outside.png")
    root.mkdir()
    (root / "escape.png").symlink_to(tmp_path / "outside.png")
    data = _payload()
    data["images"][0]["file_name"] = "escape.png"
    with pytest.raises(ValueError):
        _import(tmp_path, data, invalid_policy="quarantine")


@pytest.mark.parametrize("bad_json", [
    '{"images": [], "images": [], "annotations": [], "categories": []}',
    '{"number": NaN}', '{"number": Infinity}', '{"number": 1e999}',
    '{"images": [{"id": 1, "id": 2}]}',
])
def test_json_duplicate_keys_and_nonfinite_numbers_rejected(tmp_path, bad_json):
    from tiangong.coco_import import import_source_coco

    path = tmp_path / "bad.json"
    path.write_text(bad_json, encoding="utf-8")
    with pytest.raises(ValueError):
        import_source_coco(path, tmp_path, invalid_policy="quarantine")


def _duplicates():
    data = _payload()
    data["images"] = [dict(data["images"][0], id=i) for i in (9, 7, 8)]
    data["annotations"] = [
        _annotation(30, 9), _annotation(31, 9, category_id=1),
        _annotation(20, 7), _annotation(21, 7, category_id=1),
        _annotation(25, 8, category_id=1), _annotation(24, 8),
    ]
    return data


def test_duplicates_fail_by_default(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        _import(tmp_path, _duplicates())


def test_identical_annotation_multisets_merge_preserving_every_original_id(tmp_path):
    data = _duplicates()
    manifest = _import(tmp_path, data, duplicate_policy="merge_identical")
    sample = manifest["samples"][0]
    assert sample["sample_id"] == "image-7"
    assert sample["original_image_ids"] == [7, 8, 9]
    assert [source["original_annotation_ids"] for source in sample["sources"]] == [[20, 24, 30], [21, 25, 31]]
    assert manifest["changes"] == [{
        "action": "merge_identical", "file_name": "images/source.png",
        "image_ids": [7, 8, 9], "sample_id": "image-7",
    }]
    assert manifest["summary"]["accepted_annotations"] == 2
    data["images"].reverse()
    data["annotations"].reverse()
    reordered = _import(tmp_path, data, duplicate_policy="merge_identical")
    assert reordered["samples"] == manifest["samples"]


def test_annotation_multiplicity_is_not_erased(tmp_path):
    data = _duplicates()
    for image_id, ann_id in [(7, 22), (8, 26), (9, 32)]:
        data["annotations"].append(_annotation(ann_id, image_id))
    manifest = _import(tmp_path, data, duplicate_policy="merge_identical")
    sample = manifest["samples"][0]
    assert len(sample["sources"]) == 3
    assert sorted(aid for source in sample["sources"] for aid in source["original_annotation_ids"]) == [20, 21, 22, 24, 25, 26, 30, 31, 32]


@pytest.mark.parametrize("conflict", ["missing_annotation", "class", "metadata", "image_metadata"])
def test_conflicting_duplicate_images_quarantine_all_their_annotations(tmp_path, conflict):
    data = _duplicates()
    if conflict == "missing_annotation":
        data["annotations"].pop()
    elif conflict == "class":
        data["annotations"][0]["category_id"] = 2
    elif conflict == "metadata":
        data["annotations"][0]["review_note"] = "different export"
    else:
        data["images"][0]["license"] = 1
    with pytest.raises(ValueError):
        _import(tmp_path, data, duplicate_policy="merge_identical")
    manifest = _import(tmp_path, data, duplicate_policy="merge_identical", invalid_policy="quarantine")
    assert manifest["samples"] == []
    assert manifest["changes"] == []
    assert manifest["summary"]["review_images"] == 1
    review = manifest["review"][0]
    assert review["image_ids"] == [7, 8, 9]
    assert review["annotation_ids"] == sorted(a["id"] for a in data["annotations"])
    assert review["reason_codes"]


@pytest.mark.parametrize("updates", [
    {"bbox": [-0.16475216269992146, 2, 3, 4]},
    {"bbox": [1, -0.16919555519045965, 3, 4]},
    {"bbox": [8, 2, 3, 4]}, {"bbox": [1, 2, 3, 7]},
    {"bbox": [1, 2, 0, 4]}, {"bbox": [1, 2, -1, 4]},
    {"bbox": [1, 2, 3]}, {"bbox": [True, 2, 3, 4]},
    {"segmentation": [[1, 2, 4, 2, 5, 6]]},
    {"segmentation": [[1, 2, 1, 2, 1, 2]]},
    {"segmentation": [[1, 2, 3, 4, 5]]},
    {"segmentation": {"size": [8, 10], "counts": "abc"}},
    {"segmentation": None}, {"area": 0}, {"area": -1}, {"area": 11},
    {"iscrowd": 1}, {"ignore": 1},
])
def test_invalid_target_quarantines_whole_image_not_just_bad_source(tmp_path, updates):
    data = _payload()
    data["annotations"].append(_annotation(21, **updates))
    with pytest.raises(ValueError):
        _import(tmp_path, data)
    manifest = _import(tmp_path, data, invalid_policy="quarantine")
    assert manifest["samples"] == []
    assert manifest["review"][0]["annotation_ids"] == [20, 21]
    assert manifest["summary"]["accepted_annotations"] == 0


def test_empty_annotation_requires_explicit_allow_empty(tmp_path):
    data = _payload()
    data["annotations"] = []
    with pytest.raises(ValueError):
        _import(tmp_path, data)
    assert _import(tmp_path, data, invalid_policy="quarantine")["samples"] == []
    manifest = _import(tmp_path, data, allow_empty=True)
    assert manifest["samples"][0]["sources"] == []
    assert manifest["review"] == []


def test_bbox_only_and_clockwise_polygons_are_supported(tmp_path):
    data = _payload()
    del data["annotations"][0]["segmentation"]
    data["annotations"].append(_annotation(21, segmentation=[[1, 2, 1, 6, 4, 6, 4, 2]]))
    manifest = _import(tmp_path, data)
    assert manifest["samples"][0]["sources"][0]["segmentation"] == []
    assert len(manifest["samples"][0]["sources"][1]["segmentation"][0]) == 8


@pytest.mark.parametrize("dimensions", [(0, 8), (10, -1), (True, 8), (11, 8), (10.0, 8), (10**400, 8)])
def test_bad_declared_dimensions_quarantine_image(tmp_path, dimensions):
    data = _payload()
    data["images"][0].update(zip(("width", "height"), dimensions))
    with pytest.raises(ValueError):
        _import(tmp_path, data)
    assert _import(tmp_path, data, invalid_policy="quarantine")["samples"] == []


def test_missing_and_corrupt_png_quarantine_with_stable_reason_codes(tmp_path):
    from tiangong.coco_import import import_source_coco

    data = _payload()
    data["images"][0]["file_name"] = "missing.png"
    manifest = _import(tmp_path, data, invalid_policy="quarantine")
    assert manifest["review"][0]["reason_codes"] == ["invalid_png"]
    (tmp_path / "data/missing.png").write_bytes(b"not a PNG")
    manifest = import_source_coco(tmp_path / "result.json", tmp_path / "data", invalid_policy="quarantine")
    assert manifest["review"][0]["reason_codes"] == ["invalid_png"]


def test_identical_bytes_under_different_names_keep_identity_and_digest(tmp_path):
    data = _payload()
    data["images"].append(dict(data["images"][0], id=8, file_name="images/copy.png"))
    data["annotations"].append(_annotation(21, 8))
    _png(tmp_path / "data/images/copy.png")
    manifest = _import(tmp_path, data)
    assert [sample["sample_id"] for sample in manifest["samples"]] == ["image-7", "image-8"]
    assert len({sample["sha256"] for sample in manifest["samples"]}) == 1


@pytest.mark.parametrize("policies", [{"duplicate_policy": "first"}, {"invalid_policy": "drop"}, {"allow_empty": 1}])
def test_unrecognized_policies_reject(tmp_path, policies):
    with pytest.raises(ValueError):
        _import(tmp_path, **policies)


def test_import_does_not_modify_input_bytes(tmp_path):
    from tiangong.coco_import import import_source_coco

    _import(tmp_path)
    paths = [tmp_path / "result.json", tmp_path / "data/images/source.png"]
    before = [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths]
    import_source_coco(paths[0], tmp_path / "data")
    assert [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths] == before


def test_real_annotation_roundoff_does_not_clip_or_reject_polygon(tmp_path):
    """ann237 has py-(y+h)=8.88e-16; its geometry must remain unchanged."""
    from tiangong.coco_import import import_source_coco

    data = _payload()
    data["images"][0].update(width=150, height=150)
    polygon = [
        124.65940054495913, 7.765667574931881, 125.68119891008173, 5.313351498637602,
        124.04632152588556, 3.2697547683923704, 122.0027247956403, 3.065395095367848,
        120.1634877384196, 3.6784741144414173, 119.95912806539509, 5.313351498637602,
        120.98092643051771, 6.743869209809264,
    ]
    data["annotations"] = [_annotation(
        237, bbox=[119.95912806539509, 3.065395095367848, 5.722070844686641, 4.700272479564033],
        category_id=1, area=18.918582809286818, segmentation=[polygon],
    )]
    _png(tmp_path / "images/source.png", width=150, height=150)
    path = tmp_path / "result.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    manifest = import_source_coco(path, tmp_path)
    assert manifest["samples"][0]["sources"][0]["segmentation"] == [polygon]


def test_coco_fifo_is_rejected_without_waiting_for_a_writer(tmp_path):
    path = tmp_path / "blocked.json"
    os.mkfifo(path)
    script = """\
import sys
from tiangong.coco_import import import_source_coco
try:
    import_source_coco(sys.argv[1], sys.argv[2])
except ValueError:
    raise SystemExit(0)
raise SystemExit(1)
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", script, str(path), str(tmp_path)],
            capture_output=True, text=True, timeout=5,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("COCO importer blocked on a FIFO instead of rejecting non-regular input")
    assert result.returncode == 0, result.stderr


def test_coco_oversize_input_is_rejected_before_json_decoding(tmp_path):
    from tiangong.coco_import import import_source_coco

    path = tmp_path / "huge.json"
    with path.open("wb") as stream:
        stream.seek(64 * 1024 * 1024)
        stream.write(b" ")
    with pytest.raises(ValueError, match="64 MiB"):
        import_source_coco(path, tmp_path)


def test_nonzero_area_self_crossing_polygon_rejects_or_quarantines_entire_image(tmp_path):
    data = _payload()
    data["annotations"].append(_annotation(
        21, bbox=[0, 0, 4, 4], segmentation=[[0, 0, 4, 4, 0, 4, 3, 0]], area=2,
    ))
    with pytest.raises(ValueError, match="invalid_segmentation"):
        _import(tmp_path, data)
    manifest = _import(tmp_path, data, invalid_policy="quarantine")
    assert manifest["samples"] == []
    assert manifest["review"][0]["annotation_ids"] == [20, 21]
    assert manifest["review"][0]["reason_codes"] == ["invalid_segmentation"]


@pytest.mark.parametrize("polygon", [
    [1, 2, 2, 2, 4, 2, 4, 6, 1, 6],
    [1, 2, 4, 2, 4, 6, 1, 6, 1, 2],
])
def test_import_preserves_collinear_vertices_and_explicit_closure(tmp_path, polygon):
    data = _payload()
    data["annotations"][0]["segmentation"] = [polygon]
    manifest = _import(tmp_path, data)
    assert manifest["samples"][0]["sources"][0]["segmentation"] == [polygon]


def test_deeply_nested_coco_json_reports_value_error(tmp_path):
    from tiangong.coco_import import import_source_coco

    path = tmp_path / "nested.json"
    path.write_text('{"metadata":' + '[' * 1100 + '0' + ']' * 1100 + '}', encoding="utf-8")
    with pytest.raises(ValueError, match="nesting depth"):
        import_source_coco(path, tmp_path)


@pytest.mark.parametrize("real_png_bytes", [True, False])
def test_unsupported_extension_quarantines_only_affected_image(tmp_path, real_png_bytes):
    data = _payload()
    data["images"].append(dict(data["images"][0], id=8, file_name="images/unsupported.jpg"))
    data["annotations"].append(_annotation(21, image_id=8))
    unsupported = tmp_path / "data/images/unsupported.jpg"
    if real_png_bytes:
        _png(unsupported)
    else:
        unsupported.parent.mkdir(parents=True)
        unsupported.write_bytes(b"not a valid PNG")
    with pytest.raises(ValueError, match="unsupported_image_extension"):
        _import(tmp_path, data)
    manifest = _import(tmp_path, data, invalid_policy="quarantine")
    assert [sample["sample_id"] for sample in manifest["samples"]] == ["image-7"]
    assert manifest["review"] == [{
        "file_name": "images/unsupported.jpg", "image_ids": [8], "annotation_ids": [21],
        "reason_codes": ["unsupported_image_extension"],
    }]
    assert manifest["summary"]["accepted_images"] == 1
    assert manifest["summary"]["review_images"] == 1
