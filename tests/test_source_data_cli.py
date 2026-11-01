"""Real CLI subprocesses exercise portable source preparation and splitting."""

import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import zlib

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]


def _run(*arguments, timeout=90):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPOSITORY / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "tiangong.source_cli", *map(str, arguments)],
        env=env, capture_output=True, text=True, timeout=timeout, check=False,
    )


def _png(path, color):
    def chunk(kind, content):
        return struct.pack(">I", len(content)) + kind + content + struct.pack(
            ">I", zlib.crc32(content, zlib.crc32(kind)),
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 10, 8, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress((b"\x00" + bytes(color) * 10) * 8))
        + chunk(b"IEND", b"")
    )


def _input(tmp_path, *, empty=False, duplicate=False, invalid=False):
    root = tmp_path / "images"
    images, annotations = [], []
    for index, color in enumerate(((255, 0, 0), (0, 255, 0), (0, 0, 255)), start=1):
        _png(root / f"{index}.png", color)
        images.append({"id": index, "file_name": f"{index}.png", "width": 10, "height": 8})
        if not empty:
            annotations.append({
                "id": index, "image_id": index, "category_id": index - 1,
                "bbox": [1, 2, 3, 4], "area": 12, "segmentation": [], "iscrowd": 0,
            })
    if duplicate:
        images.append(dict(images[0], id=4))
        annotations.append(dict(annotations[0], id=4, image_id=4))
    if invalid:
        annotations[-1]["bbox"] = [-1, 2, 3, 4]
    payload = {
        "images": images, "annotations": annotations,
        "categories": [{"id": i, "name": name} for i, name in enumerate(("CJ", "CS", "FRI", "FRII"))],
    }
    coco = tmp_path / "result.json"
    coco.write_text(json.dumps(payload), encoding="utf-8")
    return coco, root


def _prepare(tmp_path, **options):
    coco, root = _input(tmp_path, **options)
    manifest = tmp_path / "manifest.json"
    result = _run("prepare", "--coco", coco, "--image-root", root, "--output", manifest)
    assert result.returncode == 0, result.stderr
    return manifest, root, result


def test_prepare_inspect_split_and_dataset_from_relocated_root(tmp_path):
    manifest, root, prepared = _prepare(tmp_path)
    summary = json.loads(prepared.stdout)
    assert summary["summary"]["accepted_images"] == 3
    assert summary["partition"]["status"] == "unassigned"
    inspected = _run("inspect", "--manifest", manifest)
    assert inspected.returncode == 0, inspected.stderr
    assert json.loads(inspected.stdout)["summary"]["accepted_annotations"] == 3

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assignments = tmp_path / "assignments.json"
    assignments.write_text(json.dumps([
        {"sample_id": sample["sample_id"], "group_id": f"physical-{i}", "split": split}
        for i, (sample, split) in enumerate(zip(payload["samples"], ("train", "val", "test")))
    ]), encoding="utf-8")
    split_manifest = tmp_path / "assigned.json"
    split = _run(
        "split", "--manifest", manifest, "--assignments", assignments,
        "--grouping-provenance", "Synthetic integration fixture; no sky identity claim",
        "--output", split_manifest,
    )
    assert split.returncode == 0, split.stderr
    assert json.loads(split.stdout)["partition"]["status"] == "assigned"
    assert json.loads(manifest.read_text(encoding="utf-8"))["partition"]["status"] == "unassigned"

    moved = tmp_path / "relocated"
    root.rename(moved)
    from tiangong.source_data import SourceDataset

    dataset = SourceDataset(split_manifest, moved, split="train")
    image, target, sample_id = dataset[0]
    assert len(dataset) == 1
    assert tuple(image.shape) == (3, 8, 10)
    assert target["labels"].tolist() == [0]
    assert target["pixel_boxes"].tolist() == [[1, 2, 4, 6]]
    assert sample_id == payload["samples"][0]["sample_id"]


def test_prepare_refuses_overwrite_and_preserves_original_bytes(tmp_path):
    manifest, root, _ = _prepare(tmp_path)
    original = manifest.read_bytes()
    result = _run("prepare", "--coco", tmp_path / "result.json", "--image-root", root, "--output", manifest)
    assert result.returncode != 0
    assert result.stdout == ""
    assert "error" in result.stderr.lower()
    assert "Traceback" not in result.stderr
    assert manifest.read_bytes() == original


@pytest.mark.parametrize("bad_input", ["missing", "malformed", "unsafe"])
def test_prepare_invalid_input_returns_error_without_output(tmp_path, bad_input):
    coco, root = _input(tmp_path)
    if bad_input == "missing":
        coco.unlink()
    elif bad_input == "malformed":
        coco.write_text("{", encoding="utf-8")
    else:
        payload = json.loads(coco.read_text(encoding="utf-8"))
        payload["images"][0]["file_name"] = "../escape.png"
        coco.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "bad.json"
    result = _run("prepare", "--coco", coco, "--image-root", root, "--output", output)
    assert result.returncode != 0
    assert result.stdout == ""
    assert "error" in result.stderr.lower()
    assert "Traceback" not in result.stderr
    assert not output.exists()


def test_prepare_passes_explicit_duplicate_quarantine_and_empty_policies(tmp_path):
    coco, root = _input(tmp_path, duplicate=True, invalid=True)
    result = _run(
        "prepare", "--coco", coco, "--image-root", root, "--output", tmp_path / "review.json",
        "--duplicate-policy", "merge_identical", "--invalid-policy", "quarantine",
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["summary"]["review_images"] > 0
    coco, root = _input(tmp_path, empty=True)
    result = _run(
        "prepare", "--coco", coco, "--image-root", root, "--output", tmp_path / "empty.json", "--allow-empty",
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["summary"]["accepted_images"] == 3
    assert json.loads(result.stdout)["summary"]["accepted_annotations"] == 0


@pytest.mark.parametrize("assignment_payload", [{}, [], [{"sample_id": "unknown", "group_id": "g", "split": "train"}]])
def test_split_rejects_noncovering_assignments_without_writing(tmp_path, assignment_payload):
    manifest, _, _ = _prepare(tmp_path)
    assignments = tmp_path / "assignments.json"
    assignments.write_text(json.dumps(assignment_payload), encoding="utf-8")
    output = tmp_path / "assigned.json"
    result = _run(
        "split", "--manifest", manifest, "--assignments", assignments,
        "--grouping-provenance", "test fixture", "--output", output,
    )
    assert result.returncode != 0
    assert result.stdout == ""
    assert "Traceback" not in result.stderr
    assert not output.exists()


def test_inspect_invalid_manifest_is_not_reported_as_success(tmp_path):
    manifest = tmp_path / "bad.json"
    manifest.write_text('{"schema_version": 999}', encoding="utf-8")
    result = _run("inspect", "--manifest", manifest)
    assert result.returncode != 0
    assert result.stdout == ""
    assert "error" in result.stderr.lower()
    assert "Traceback" not in result.stderr


def test_split_rejects_duplicate_json_keys_in_assignment_file(tmp_path):
    manifest, _, _ = _prepare(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    rows = [
        {"sample_id": sample["sample_id"], "group_id": f"g{i}", "split": split}
        for i, (sample, split) in enumerate(zip(payload["samples"], ("train", "val", "test")))
    ]
    assignments = tmp_path / "ambiguous.json"
    assignments.write_text(
        json.dumps(rows).replace('"group_id": "g0"', '"group_id": "hidden", "group_id": "g0"'),
        encoding="utf-8",
    )
    output = tmp_path / "assigned.json"
    result = _run(
        "split", "--manifest", manifest, "--assignments", assignments,
        "--grouping-provenance", "test fixture", "--output", output,
    )
    assert result.returncode != 0
    assert "duplicate" in result.stderr.lower()
    assert not output.exists()


@pytest.mark.parametrize("bad_file", ["fifo", "directory", "oversized", "utf8", "nan", "infinity", "exponent", "deep"])
def test_split_assignment_reader_is_bounded_and_strict(tmp_path, bad_file):
    manifest, _, _ = _prepare(tmp_path)
    assignments = tmp_path / "bad-assignments.json"
    if bad_file == "fifo":
        if not hasattr(os, "mkfifo"):
            pytest.skip("FIFO unavailable on this platform")
        os.mkfifo(assignments)
    elif bad_file == "directory":
        assignments.mkdir()
    elif bad_file == "oversized":
        with assignments.open("wb") as handle:
            handle.write(b"[]")
            handle.truncate(64 * 1024 * 1024 + 1)
    elif bad_file == "utf8":
        assignments.write_bytes(b"\xff\xfe")
    elif bad_file == "deep":
        assignments.write_text("[" * 1100 + "]" * 1100, encoding="utf-8")
    else:
        number = {"nan": "NaN", "infinity": "Infinity", "exponent": "1e999"}[bad_file]
        assignments.write_text("[" + number + "]", encoding="utf-8")
    output = tmp_path / "assigned.json"
    try:
        result = _run(
            "split", "--manifest", manifest, "--assignments", assignments,
            "--grouping-provenance", "test fixture", "--output", output, timeout=3,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("assignment reader blocked on a nonregular or invalid input")
    assert result.returncode != 0
    assert "Traceback" not in result.stderr
    if bad_file in {"fifo", "directory"}:
        assert "regular" in result.stderr.lower()
    elif bad_file == "oversized":
        assert "64 MiB" in result.stderr
    elif bad_file in {"nan", "infinity", "exponent"}:
        assert "finite" in result.stderr
    assert not output.exists()


def test_help_works_without_loading_optional_numeric_or_image_dependencies():
    # -S removes site packages; the package version is the only metadata needed
    # before CLI argument parsing. The dependency guard catches eager imports.
    code = """
import importlib.abc
import importlib.metadata
import runpy
import sys
importlib.metadata.version = lambda distribution: 'test'
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'PIL'}:
            raise RuntimeError('eager optional import: ' + fullname)
sys.meta_path.insert(0, Guard())
sys.argv = ['tiangong-source-data', '--help']
runpy.run_module('tiangong.source_cli', run_name='__main__')
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPOSITORY / "src")
    result = subprocess.run([sys.executable, "-S", "-c", code], env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "prepare" in result.stdout and "inspect" in result.stdout and "split" in result.stdout
