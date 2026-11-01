"""验证安装入口的显式checkpoint迁移、检查和防覆盖。"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from tiangong.model import MorphologyCNN
from tiangong.detection_data import PREPROCESS_VERSION


def run_cli(*args):
    command = Path(sys.executable).parent / "tiangong-bundle"
    assert command.is_file(), "缺少安装入口 tiangong-bundle"
    return subprocess.run([str(command), *map(str, args)], capture_output=True, text=True, timeout=90)


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save(dict(model=MorphologyCNN(2).state_dict(), classes=["FR I", "FR II"],
                    size=16, seed=7, epoch=1, manifest_sha256="a" * 64), path)
    return path


def test_installed_cli_migrates_inspects_and_refuses_overwrite(checkpoint, tmp_path):
    path = tmp_path / "result.tgb"
    args = ["migrate", "--checkpoint", checkpoint, "--output", path,
            "--task", "morphology_classification", "--source-description", "known local project checkpoint",
            "--legacy-preprocess-version", PREPROCESS_VERSION, "--trust-checkpoint"]
    result = run_cli(*args)
    assert result.returncode == 0, result.stderr
    before = path.read_bytes()
    result = run_cli("inspect", "--bundle", path)
    assert result.returncode == 0, result.stderr
    metadata = json.loads(result.stdout)
    assert metadata["labels"]["classes"] == ["FR I", "FR II"]
    assert metadata["provenance"]["migration"]["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    result = run_cli(*args)
    assert result.returncode == 2 and path.read_bytes() == before


def test_cli_does_not_guess_legacy_preprocessing(checkpoint, tmp_path):
    path = tmp_path / "unconfirmed.tgb"
    result = run_cli("migrate", "--checkpoint", checkpoint, "--output", path,
                     "--task", "morphology_classification", "--source-description", "legacy", "--trust-checkpoint")
    assert result.returncode == 2 and "预处理" in result.stderr
    assert not path.exists()


def test_cli_refuses_checkpoint_without_explicit_trust(checkpoint, tmp_path):
    path = tmp_path / "not-trusted.tgb"
    result = run_cli("migrate", "--checkpoint", checkpoint, "--output", path,
                     "--task", "morphology_classification", "--source-description", "legacy",
                     "--legacy-preprocess-version", PREPROCESS_VERSION)
    assert result.returncode == 2 and "trust" in result.stderr
    assert not path.exists()
