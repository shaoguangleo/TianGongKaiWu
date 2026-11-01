#!/usr/bin/env python3
"""Collect local/server evidence without installing or guessing GPU dependencies.

Run this script with the interpreter that owns the intended environment. CUDA
acceptance is optional and requires actual device execution; a CPU/macOS run
does not establish Rocky/V100 compatibility or scientific model validity.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time


SOURCE_TESTS = (
    "tests/test_png_source.py",
    "tests/test_coco_import.py",
    "tests/test_source_manifest.py",
    "tests/test_source_geometry.py",
    "tests/test_source_data_cli.py",
    "tests/test_server_validation.py",
)


def _nvidia_smi() -> dict:
    """Probe driver/name using NVIDIA's optional CLI; bound a broken driver wait."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"status": "unavailable", "error": "nvidia-smi not found on PATH"}
    command = [executable, "--query-gpu=driver_version,name", "--format=csv,noheader"]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=10)
        return {
            "status": "passed" if completed.returncode == 0 else "failed",
            "command": command, "returncode": completed.returncode,
            "stdout": completed.stdout, "stderr": completed.stderr,
        }
    except subprocess.TimeoutExpired as exc:
        # TimeoutExpired can expose bytes even when text=True was requested.
        def decoded(value):
            return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value

        return {
            "status": "failed", "command": command,
            "error": f"TimeoutExpired: {exc}",
            "stdout": decoded(exc.stdout), "stderr": decoded(exc.stderr),
        }
    except OSError as exc:
        return {"status": "failed", "command": command, "error": f"{type(exc).__name__}: {exc}"}


def collect_environment() -> dict:
    """Record import/runtime information, including failed dependency imports."""
    environment = {
        "python_version": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "nvidia_smi": _nvidia_smi(),
    }
    if platform.system() == "Linux":
        try:
            environment["linux_distribution"] = platform.freedesktop_os_release()
        except OSError as exc:
            environment["linux_distribution_error"] = str(exc)
    try:
        import PIL

        environment["pillow"] = {"available": True, "version": PIL.__version__}
    except Exception as exc:
        environment["pillow"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        environment["torch"] = {
            "available": True,
            "version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "cuda_available": cuda_available,
            "arch_list": torch.cuda.get_arch_list(),
            "devices": [
                {"index": index, "name": torch.cuda.get_device_name(index),
                 "capability": list(torch.cuda.get_device_capability(index))}
                for index in range(torch.cuda.device_count())
            ] if cuda_available else [],
        }
    except Exception as exc:
        environment["torch"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return environment


def check_cuda() -> dict:
    """Execute a real tensor matmul and backward pass; require finite results."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {"status": "failed", "error": "CUDA unavailable in the selected Python environment"}
        device = torch.device("cuda:0")
        value = torch.randn(16, 16, device=device, dtype=torch.float32, requires_grad=True)
        loss = (value @ value.transpose(0, 1)).square().mean()
        loss.backward()
        torch.cuda.synchronize(device)
        if value.grad is None or not torch.isfinite(loss).item() or not torch.isfinite(value.grad).all().item():
            return {"status": "failed", "error": "CUDA matmul/backward produced missing or nonfinite values"}
        return {
            "status": "passed",
            "device": str(device),
            "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "operation": "float32 matmul, squared mean loss, backward; finite loss and gradients",
            "loss": float(loss.detach().cpu().item()),
        }
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}


def _write_json(path: Path, payload: dict) -> None:
    """Evidence files are created exclusively inside a newly created directory."""
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New evidence directory; existing paths are rejected")
    parser.add_argument("--suite", choices=("source", "full"), default="source")
    parser.add_argument("--require-cuda", action="store_true", help="Require a real finite CUDA forward/backward pass")
    args = parser.parse_args(argv)
    output = args.output_dir
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        print(f"validation error: cannot create new evidence directory: {exc}", file=sys.stderr)
        return 2

    repository = Path(__file__).resolve().parents[1]
    started = time.monotonic()
    result = {
        "status": "failed",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "suite": args.suite,
        "require_cuda": args.require_cuda,
        "cuda": {"status": "not_requested"},
    }
    try:
        environment = collect_environment()
        _write_json(output / "environment.json", environment)
        command = [sys.executable, "-m", "pytest", "-q"]
        if args.suite == "source":
            command.extend(SOURCE_TESTS)
        result["pytest"] = {"command": command, "cwd": str(repository), "returncode": None}
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repository / "src") + os.pathsep + env.get("PYTHONPATH", "")
        with (output / "pytest.log").open("x", encoding="utf-8") as log:
            try:
                completed = subprocess.run(
                    command, cwd=repository, env=env, stdout=log,
                    stderr=subprocess.STDOUT, text=True, check=False,
                )
                result["pytest"]["returncode"] = completed.returncode
            except OSError as exc:
                result["pytest"]["error"] = f"{type(exc).__name__}: {exc}"
                log.write(f"\nUnable to start pytest: {result['pytest']['error']}\n")

        if args.require_cuda:
            if not environment["torch"].get("cuda_available", False):
                result["cuda"] = {"status": "failed", "error": "CUDA unavailable in the selected Python environment"}
            else:
                result["cuda"] = check_cuda()

        dependencies_ok = all(environment[name].get("available", False) for name in ("torch", "pillow"))
        if dependencies_ok and result["pytest"]["returncode"] == 0 and result["cuda"]["status"] != "failed":
            result["status"] = "passed"
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        _write_json(output / "result.json", result)
    except (OSError, ValueError) as exc:
        # A read-only/full filesystem cannot always retain further evidence.
        # Keep all files already written and report the write failure explicitly.
        print(f"validation error: unable to complete evidence files: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    if result["status"] == "passed":
        return 0
    print(f"validation failed; inspect evidence at {output}", file=sys.stderr)
    pytest_returncode = result["pytest"]["returncode"]
    return pytest_returncode if isinstance(pytest_returncode, int) and 0 < pytest_returncode < 126 else 1


if __name__ == "__main__":
    raise SystemExit(main())
