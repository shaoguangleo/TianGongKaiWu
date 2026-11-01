"""Server evidence handling; external pytest runs are replaced to avoid recursion."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_source_data.py"


def _module():
    assert SCRIPT.is_file(), "The standalone server-validation entrypoint is missing"
    spec = importlib.util.spec_from_file_location("source_server_validation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _environment():
    return {
        "python_version": "3.12.fixture", "executable": sys.executable,
        "platform": "test-fixture", "os": "test-fixture", "machine": "test-fixture",
        "torch": {"available": True, "version": "fixture", "cuda_version": None,
                  "cuda_available": False, "devices": [], "arch_list": []},
        "pillow": {"available": True, "version": "fixture"},
    }


def _fake_process(monkeypatch, module, *, returncode=0, error=None):
    # Only replace the external process boundary. Real output files, return-code
    # decisions, JSON persistence, argv construction, and error handling execute.
    def run(command, **kwargs):
        assert command[:4] == [sys.executable, "-m", "pytest", "-q"]
        assert Path(kwargs["cwd"]) == SCRIPT.parents[1]
        kwargs["stdout"].write("fixture pytest transcript\n")
        kwargs["stdout"].flush()
        if error is not None:
            raise error
        return subprocess.CompletedProcess(command, returncode)

    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(module, "collect_environment", _environment)


@pytest.mark.parametrize("suite", ["source", "full"])
def test_server_records_environment_logs_and_current_python_command(tmp_path, monkeypatch, suite):
    module = _module()
    _fake_process(monkeypatch, module)
    output = tmp_path / "evidence"
    assert module.main(["--output-dir", str(output), "--suite", suite]) == 0
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    environment = json.loads((output / "environment.json").read_text(encoding="utf-8"))
    assert result["status"] == "passed"
    assert result["pytest"]["returncode"] == 0
    assert result["cuda"]["status"] == "not_requested"
    command = result["pytest"]["command"]
    assert command[:4] == [sys.executable, "-m", "pytest", "-q"]
    if suite == "source":
        assert "tests/test_source_data_cli.py" in command
        assert "tests/test_png_source.py" in command
        assert "tests/test_server_validation.py" in command
        assert "tests/test_coco_import.py" in command
        assert "tests/test_source_manifest.py" in command
        assert "tests/test_source_geometry.py" in command
    else:
        assert command == [sys.executable, "-m", "pytest", "-q"]
    assert environment["python_version"] == "3.12.fixture"
    assert environment["torch"]["cuda_available"] is False
    assert (output / "pytest.log").read_text(encoding="utf-8") == "fixture pytest transcript\n"


def test_server_preserves_pytest_failure_and_returns_nonzero(tmp_path, monkeypatch):
    module = _module()
    _fake_process(monkeypatch, module, returncode=5)
    output = tmp_path / "failure"
    assert module.main(["--output-dir", str(output)]) != 0
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert result["pytest"]["returncode"] == 5
    assert "fixture pytest transcript" in (output / "pytest.log").read_text(encoding="utf-8")


def test_server_preserves_spawn_error_as_failed_result(tmp_path, monkeypatch):
    module = _module()
    _fake_process(monkeypatch, module, error=OSError("simulated spawn denial"))
    output = tmp_path / "spawn-error"
    assert module.main(["--output-dir", str(output)]) != 0
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert "simulated spawn denial" in result["pytest"]["error"]
    assert "simulated spawn denial" in (output / "pytest.log").read_text(encoding="utf-8")
    assert (output / "environment.json").is_file()


def test_server_rejects_existing_output_directory_without_modifying_it(tmp_path, capsys):
    module = _module()
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("unchanged", encoding="utf-8")
    assert module.main(["--output-dir", str(output)]) != 0
    assert {path.name for path in output.iterdir()} == {"keep.txt"}
    assert (output / "keep.txt").read_text(encoding="utf-8") == "unchanged"
    assert "Traceback" not in capsys.readouterr().err


def test_server_reports_output_creation_denial_without_traceback(tmp_path, monkeypatch, capsys):
    module = _module()

    def denied(*args, **kwargs):
        raise PermissionError("simulated read-only output")

    monkeypatch.setattr(module.Path, "mkdir", denied)
    assert module.main(["--output-dir", str(tmp_path / "denied")]) != 0
    stderr = capsys.readouterr().err
    assert "simulated read-only output" in stderr
    assert "Traceback" not in stderr


def test_server_require_cuda_fails_and_keeps_pytest_logs_when_unavailable(tmp_path, monkeypatch):
    module = _module()
    _fake_process(monkeypatch, module)
    # CUDA is demonstrably absent in this fixture; the real request gating must
    # reject before trying to manufacture a device-level success.
    output = tmp_path / "no-gpu"
    assert module.main(["--output-dir", str(output), "--require-cuda"]) != 0
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert result["cuda"]["status"] == "failed"
    assert "unavailable" in result["cuda"]["error"].lower()
    assert result["pytest"]["returncode"] == 0
    assert (output / "pytest.log").is_file()


def test_missing_required_dependency_is_recorded_and_fails(tmp_path, monkeypatch):
    module = _module()
    _fake_process(monkeypatch, module)
    environment = _environment()
    environment["pillow"] = {"available": False, "error": "not installed"}
    monkeypatch.setattr(module, "collect_environment", lambda: environment)
    output = tmp_path / "missing-pillow"
    assert module.main(["--output-dir", str(output)]) != 0
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    recorded = json.loads((output / "environment.json").read_text(encoding="utf-8"))
    assert recorded["pillow"]["error"] == "not installed"


def test_server_help_uses_only_standard_library():
    assert SCRIPT.is_file(), "The standalone server-validation entrypoint is missing"
    result = subprocess.run([sys.executable, "-S", str(SCRIPT), "--help"], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "--require-cuda" in result.stdout


@pytest.mark.parametrize("failure", [None, "timeout", "spawn", "exit"])
def test_environment_records_driver_probe_success_or_failure(monkeypatch, failure):
    module = _module()
    # Probe logic uses real Python/OS metadata, but the machine-specific
    # executables and optional modules are replaced at their external boundary.
    monkeypatch.setitem(sys.modules, "PIL", SimpleNamespace(__version__="pillow-fixture"))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        __version__="torch-fixture", version=SimpleNamespace(cuda=None),
        cuda=SimpleNamespace(is_available=lambda: False, get_arch_list=lambda: []),
    ))
    import shutil

    monkeypatch.setattr(shutil, "which", lambda program: "/fixture/nvidia-smi")
    real_run = subprocess.run

    def probe(command, **kwargs):
        if command[0] != "/fixture/nvidia-smi":
            return real_run(command, **kwargs)
        assert command == ["/fixture/nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"]
        assert 0 < kwargs["timeout"] <= 30
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"], output="partial driver output")
        if failure == "spawn":
            raise OSError("simulated driver probe denial")
        if failure == "exit":
            return subprocess.CompletedProcess(command, 9, "", "driver communication failed")
        return subprocess.CompletedProcess(command, 0, "fixture-driver, fixture-GPU\n", "")

    monkeypatch.setattr(module.subprocess, "run", probe)
    environment = module.collect_environment()
    assert environment["python_version"]
    assert environment["torch"]["version"] == "torch-fixture"
    driver = environment["nvidia_smi"]
    if failure is None:
        assert driver["status"] == "passed"
        assert driver["stdout"] == "fixture-driver, fixture-GPU\n"
    else:
        assert driver["status"] == "failed"
        if failure == "timeout":
            assert "TimeoutExpired" in driver["error"]
            assert driver["stdout"] == "partial driver output"
        elif failure == "spawn":
            assert "simulated driver probe denial" in driver["error"]
        else:
            assert driver["returncode"] == 9
            assert driver["stderr"] == "driver communication failed"


def test_environment_records_absent_driver_utility(monkeypatch):
    module = _module()
    monkeypatch.setitem(sys.modules, "PIL", SimpleNamespace(__version__="pillow-fixture"))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        __version__="torch-fixture", version=SimpleNamespace(cuda=None),
        cuda=SimpleNamespace(is_available=lambda: False, get_arch_list=lambda: []),
    ))
    import shutil

    monkeypatch.setattr(shutil, "which", lambda program: None)
    driver = module.collect_environment()["nvidia_smi"]
    assert driver["status"] == "unavailable"
    assert "not found" in driver["error"]
