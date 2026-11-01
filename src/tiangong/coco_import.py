"""Read-only COCO import for complete radio sources, with explicit quarantine.

Identity errors are fatal even under quarantine. Image-local failures retain all
original image and annotation IDs in review; no bad target is silently dropped.
"""

from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat

from tiangong.png import read_png
from tiangong.source_geometry import validate_simple_polygon


_CATEGORY_NAMES = ("CJ", "CS", "FRI", "FRII")
MAX_COCO_BYTES = 64 * 1024 * 1024


class _ImageInvalid(ValueError):
    """An image-local failure with a portable, non-sensitive reason code."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"COCO JSON contains duplicate key: {key}")
        result[key] = value
    return result


def _finite_json_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("COCO JSON contains non-finite number")
    return number


def _reject_constant(value: str) -> None:
    raise ValueError(f"COCO JSON contains non-finite constant: {value}")


def _id(value: object, context: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{context} must be a non-negative integer")
    return value


def _relative_image_path(value: object, root: Path) -> Path:
    """Reject ambiguous path spellings and all resolved symlink escapes."""
    if (
        not isinstance(value, str) or not value or "\\" in value or ":" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
        or PurePosixPath(value).is_absolute()
        or any(part in ("", ".", "..") for part in value.split("/"))
    ):
        raise ValueError("COCO file_name must be a safe relative POSIX path")
    path = root / value
    try:
        path.resolve().relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("COCO image path escapes image_root or has invalid links") from exc
    return path


def _records(payload: dict, field: str) -> list[dict]:
    records = payload.get(field)
    if not isinstance(records, list) or any(not isinstance(item, dict) for item in records):
        raise ValueError(f"COCO {field} must be a list of objects")
    return records


def _signature(record: dict, *, excluded: tuple[str, ...]) -> str:
    # All remaining fields participate, including extra COCO export metadata.
    return json.dumps(
        {key: value for key, value in record.items() if key not in excluded},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    )


def _number(value: object, reason: str) -> float:
    if type(value) not in (int, float):
        raise _ImageInvalid(reason)
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise _ImageInvalid(reason) from exc
    if not math.isfinite(number):
        raise _ImageInvalid(reason)
    return number


def _source(annotation: dict, width: int, height: int, originals: list[int]) -> dict:
    for field in ("iscrowd", "ignore"):
        flag = annotation.get(field, 0)
        if type(flag) is not int or flag != 0:
            raise _ImageInvalid(f"unsupported_{field}")
    bbox = annotation.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise _ImageInvalid("invalid_bbox")
    x, y, box_width, box_height = [_number(value, "invalid_bbox") for value in bbox]
    xmax, ymax = x + box_width, y + box_height
    if (
        box_width <= 0 or box_height <= 0 or x < 0 or y < 0
        or not math.isfinite(xmax) or not math.isfinite(ymax)
        or xmax > width or ymax > height or xmax <= x or ymax <= y
    ):
        raise _ImageInvalid("invalid_bbox")
    area = _number(annotation.get("area"), "invalid_area")
    if area <= 0:
        raise _ImageInvalid("invalid_area")
    segmentation = annotation.get("segmentation", [])
    if isinstance(segmentation, dict):
        raise _ImageInvalid("unsupported_segmentation")
    if not isinstance(segmentation, list):
        raise _ImageInvalid("invalid_segmentation")
    polygon_areas = []
    for polygon in segmentation:
        if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2:
            raise _ImageInvalid("invalid_segmentation")
        vertices = [_number(value, "invalid_segmentation") for value in polygon]
        points = list(zip(vertices[::2], vertices[1::2]))
        try:
            validate_simple_polygon(points)
        except ValueError as exc:
            raise _ImageInvalid("invalid_segmentation") from exc
        for px, py in points:
            if not (0 <= px <= width and 0 <= py <= height):
                raise _ImageInvalid("invalid_segmentation")
            # Permit only floating arithmetic roundoff in x + box_width; never
            # clip a vertex or use this tolerance for the image boundary.
            tolerance = 1e-12 * max(width, height, 1)
            if px < x - tolerance or px > xmax + tolerance or py < y - tolerance or py > ymax + tolerance:
                raise _ImageInvalid("invalid_segmentation")
        polygon_area = abs(math.fsum(
            px * points[(index + 1) % len(points)][1]
            - points[(index + 1) % len(points)][0] * py
            for index, (px, py) in enumerate(points)
        )) / 2
        if not math.isfinite(polygon_area) or polygon_area <= 0:
            raise _ImageInvalid("invalid_segmentation")
        polygon_areas.append(polygon_area)
    if polygon_areas and not math.isclose(math.fsum(polygon_areas), area, rel_tol=1e-6, abs_tol=1e-6):
        raise _ImageInvalid("invalid_area")
    return {
        "source_id": f"annotation-{min(originals)}",
        "category_id": annotation["category_id"], "label": annotation["category_id"],
        "bbox": [x, y, xmax, ymax], "segmentation": segmentation,
        "area": area, "original_annotation_ids": sorted(originals),
    }


def _merge_targets(images: list[dict], annotations_by_image: dict[int, list[dict]]) -> list[tuple[dict, list[int]]]:
    """Match equal annotation multisets, preserving duplicate multiplicity."""
    image_signatures = {_signature(image, excluded=("id",)) for image in images}
    if len(image_signatures) != 1:
        raise _ImageInvalid("conflicting_image_metadata")
    grouped = []
    for image in images:
        by_signature = defaultdict(list)
        for annotation in annotations_by_image[image["id"]]:
            by_signature[_signature(annotation, excluded=("id", "image_id"))].append(annotation)
        for records in by_signature.values():
            records.sort(key=lambda record: record["id"])
        grouped.append(by_signature)
    counts = [Counter({key: len(records) for key, records in group.items()}) for group in grouped]
    if any(count != counts[0] for count in counts[1:]):
        raise _ImageInvalid("conflicting_annotations")
    targets = []
    for signature, records in grouped[0].items():
        for index, annotation in enumerate(records):
            originals = [group[signature][index]["id"] for group in grouped]
            targets.append((annotation, sorted(originals)))
    return sorted(targets, key=lambda target: min(target[1]))


def import_source_coco(
    coco_path: str | Path,
    image_root: str | Path,
    *,
    duplicate_policy: str = "error",
    invalid_policy: str = "error",
    allow_empty: bool = False,
) -> dict:
    """Import one frozen COCO byte snapshot; inputs are never modified.

    ``merge_identical`` requires identical image metadata and annotation
    multisets. ``quarantine`` isolates an entire file and every original ID if
    any target, PNG, or duplicate comparison fails. Duplicate IDs, broken
    references, unsafe paths and category errors remain fatal. Different
    filenames with equal PNG bytes retain separate identities and equal SHA256
    values, allowing the split validator to prevent cross-partition leakage.
    """
    if duplicate_policy not in ("error", "merge_identical"):
        raise ValueError("duplicate_policy must be error or merge_identical")
    if invalid_policy not in ("error", "quarantine"):
        raise ValueError("invalid_policy must be error or quarantine")
    if type(allow_empty) is not bool:
        raise ValueError("allow_empty must be boolean")
    coco_path = Path(coco_path)
    root = Path(image_root).resolve()
    if not root.is_dir():
        raise ValueError("image_root must be an existing directory")
    # O_NONBLOCK avoids waiting on a FIFO before fstat can reject it. Snapshot
    # reads are bounded even if a file grows after the initial size check.
    descriptor = os.open(coco_path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_COCO_BYTES:
            raise ValueError("COCO input must be a regular file of at most 64 MiB")
        content = stream.read(MAX_COCO_BYTES + 1)
    if len(content) > MAX_COCO_BYTES:
        raise ValueError("COCO input exceeds 64 MiB")
    try:
        payload = json.loads(
            content.decode("utf-8"), object_pairs_hook=_unique_object,
            parse_float=_finite_json_float, parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("COCO must contain valid UTF-8 JSON") from exc
    except RecursionError as exc:
        raise ValueError("COCO JSON nesting depth exceeds supported limit") from exc
    if not isinstance(payload, dict):
        raise ValueError("COCO root must be an object")
    images = _records(payload, "images")
    annotations = _records(payload, "annotations")
    categories = _records(payload, "categories")
    seen_categories = set()
    for category in categories:
        identifier = _id(category.get("id"), "category id")
        if identifier in seen_categories or identifier >= len(_CATEGORY_NAMES) or category.get("name") != _CATEGORY_NAMES[identifier]:
            raise ValueError("COCO categories must match 0:CJ, 1:CS, 2:FRI, 3:FRII exactly")
        seen_categories.add(identifier)
    if seen_categories != set(range(4)):
        raise ValueError("COCO must declare all four confirmed categories")
    image_ids, annotation_ids = set(), set()
    images_by_file = defaultdict(list)
    paths = {}
    for image in images:
        identifier = _id(image.get("id"), "image id")
        if identifier in image_ids:
            raise ValueError(f"duplicate COCO image id: {identifier}")
        image_ids.add(identifier)
        file_name = image.get("file_name")
        paths[file_name] = _relative_image_path(file_name, root)
        images_by_file[file_name].append(image)
    annotations_by_image = defaultdict(list)
    for annotation in annotations:
        identifier = _id(annotation.get("id"), "annotation id")
        if identifier in annotation_ids:
            raise ValueError(f"duplicate COCO annotation id: {identifier}")
        annotation_ids.add(identifier)
        image_id = _id(annotation.get("image_id"), "annotation image_id")
        category_id = _id(annotation.get("category_id"), "annotation category_id")
        if image_id not in image_ids or category_id not in seen_categories:
            raise ValueError(f"COCO annotation {identifier} has broken image/category reference")
        annotations_by_image[image_id].append(annotation)
    samples, review, changes = [], [], []
    for file_name, records in sorted(images_by_file.items(), key=lambda item: min(record["id"] for record in item[1])):
        records.sort(key=lambda record: record["id"])
        originals = [record["id"] for record in records]
        all_annotations = sorted(annotation["id"] for identifier in originals for annotation in annotations_by_image[identifier])
        if len(records) > 1 and duplicate_policy == "error":
            raise ValueError(f"duplicate COCO image file: {file_name}; explicitly choose merge_identical")
        try:
            if PurePosixPath(file_name).suffix.lower() != ".png":
                raise _ImageInvalid("unsupported_image_extension")
            targets = _merge_targets(records, annotations_by_image)
            width, height = records[0].get("width"), records[0].get("height")
            if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
                raise _ImageInvalid("invalid_image_dimensions")
            if not targets and not allow_empty:
                raise _ImageInvalid("empty_annotations")
            try:
                _, metadata = read_png(paths[file_name])
            except (OSError, ValueError) as exc:
                raise _ImageInvalid("invalid_png") from exc
            if (width, height) != (metadata["width"], metadata["height"]):
                raise _ImageInvalid("image_dimensions_mismatch")
            # Geometry only sees dimensions verified against the bounded PNG;
            # huge untrusted JSON integers must never enter floating arithmetic.
            sources = [_source(annotation, width, height, ids) for annotation, ids in targets]
        except _ImageInvalid as exc:
            if invalid_policy == "error":
                raise ValueError(f"COCO image {file_name}: {exc.code}") from exc
            review.append({
                "file_name": file_name, "image_ids": originals,
                "annotation_ids": all_annotations, "reason_codes": [exc.code],
            })
            continue
        sample_id = f"image-{originals[0]}"
        samples.append({
            "sample_id": sample_id, "file_name": file_name, "sha256": metadata["sha256"],
            "width": width, "height": height, "original_image_ids": originals,
            "group_id": None, "split": None, "sources": sources,
        })
        if len(records) > 1:
            changes.append({
                "action": "merge_identical", "file_name": file_name,
                "image_ids": originals, "sample_id": sample_id,
            })
    class_counts = {name: 0 for name in _CATEGORY_NAMES}
    for sample in samples:
        for source in sample["sources"]:
            class_counts[_CATEGORY_NAMES[source["category_id"]]] += 1
    manifest = {
        "schema_version": 1, "task": "source_detection",
        "annotation_unit": "complete_radio_source", "coordinate_system": "pixel_xyxy",
        "preprocessing": {"version": "png-rgb-unit-v1", "channels": 3, "alpha_policy": "require_opaque", "resize": None},
        "categories": [{"id": index, "name": name, "label": index} for index, name in enumerate(_CATEGORY_NAMES)],
        "provenance": {
            "coco_sha256": hashlib.sha256(content).hexdigest(), "coco_file_name": coco_path.name,
            "policy": {"duplicate_policy": duplicate_policy, "invalid_policy": invalid_policy, "allow_empty": allow_empty},
        },
        "partition": {"status": "unassigned", "grouping_provenance": None, "scientific_independence": "unverified"},
        "samples": samples, "review": review, "changes": changes,
        "summary": {
            "accepted_images": len(samples), "accepted_annotations": sum(class_counts.values()),
            "review_images": len(review), "annotations_by_class": class_counts,
        },
    }
    from tiangong.source_manifest import validate_source_manifest

    return validate_source_manifest(manifest)
