"""Explicit source preparation, grouping, and inspection without eager ML imports."""

import argparse
import json
import math
import os
from pathlib import Path
import stat
import sys


MAX_ASSIGNMENT_BYTES = 64 * 1024 * 1024


def _unique_assignment_keys(pairs: list[tuple[str, object]]) -> dict:
    """Do not let JSON's last-value-wins behavior hide a conflicting group."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"assignment JSON contains duplicate key: {key}")
        result[key] = value
    return result


def _finite_assignment_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("assignment JSON must contain only finite numbers")
    return number


def _reject_assignment_constant(value: str) -> None:
    raise ValueError(f"assignment JSON contains nonfinite constant: {value}")


def _read_assignments(path: Path) -> list:
    """Bound allocation and avoid waiting on FIFOs before checking file type."""
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("assignment JSON must be a regular file")
        if metadata.st_size > MAX_ASSIGNMENT_BYTES:
            raise ValueError("assignment JSON exceeds the 64 MiB size limit")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            content = handle.read(MAX_ASSIGNMENT_BYTES + 1)
        if len(content) > MAX_ASSIGNMENT_BYTES:
            raise ValueError("assignment JSON exceeds the 64 MiB size limit")
    finally:
        os.close(descriptor)
    try:
        return json.loads(
            content.decode("utf-8"), object_pairs_hook=_unique_assignment_keys,
            parse_float=_finite_assignment_float, parse_constant=_reject_assignment_constant,
        )
    except RecursionError as exc:
        raise ValueError("assignment JSON nesting is too deep") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and inspect complete-radio-source PNG/COCO manifests.")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Validate COCO and PNG data and write a new source manifest")
    prepare.add_argument("--coco", type=Path, required=True)
    prepare.add_argument("--image-root", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--duplicate-policy", choices=("error", "merge_identical"), default="error")
    prepare.add_argument("--invalid-policy", choices=("error", "quarantine"), default="error")
    prepare.add_argument("--allow-empty", action="store_true", help="Explicitly accept images with no annotations")
    split = commands.add_parser("split", help="Apply an explicit, complete sample/group/split assignment table")
    split.add_argument("--manifest", type=Path, required=True)
    split.add_argument("--assignments", type=Path, required=True, help="JSON array of sample_id, group_id, split objects")
    split.add_argument("--grouping-provenance", required=True, help="Human-provided basis for the grouping")
    split.add_argument("--output", type=Path, required=True)
    inspect = commands.add_parser("inspect", help="Validate a manifest and print counts and partition state")
    inspect.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Return zero only after validation and any requested exclusive write finish."""
    args = _parser().parse_args(argv)
    try:
        # Keep parser/help usable before Pillow, NumPy, or torch are installed.
        from .source_manifest import assign_source_splits, read_source_manifest, write_source_manifest

        if args.command == "prepare":
            from .coco_import import import_source_coco

            payload = import_source_coco(
                args.coco, args.image_root, duplicate_policy=args.duplicate_policy,
                invalid_policy=args.invalid_policy, allow_empty=args.allow_empty,
            )
            write_source_manifest(payload, args.output)
        elif args.command == "split":
            payload = read_source_manifest(args.manifest)
            assignments = _read_assignments(args.assignments)
            payload = assign_source_splits(
                payload, assignments, grouping_provenance=args.grouping_provenance,
            )
            write_source_manifest(payload, args.output)
        else:
            payload = read_source_manifest(args.manifest)

        print(json.dumps({
            "operation": args.command,
            "schema_version": payload["schema_version"],
            "task": payload["task"],
            "summary": payload["summary"],
            "partition": payload["partition"],
        }, ensure_ascii=False, sort_keys=True, allow_nan=False))
    except (OSError, ValueError, ImportError) as exc:
        print(f"source-data error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
