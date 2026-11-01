"""由安装测试复制到仓库外执行；只使用安装产物与真实 FITS 格式的模拟样本。"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import astropy
from astropy.io import fits
import numpy as np
import scipy
import safetensors
import torch

import tiangong
from tiangong.data import preprocess, read_image
from tiangong.model import MorphologyCNN
from tiangong.detection_model import TinySetDetector
from tiangong.bundle import load_bundle
from tiangong.classification_checkpoint import load_classification_checkpoint
from tiangong.detect import load_detection_checkpoint


def run_cli(name, *arguments):
    directory = "Scripts" if os.name == "nt" else "bin"
    executable = Path(sys.prefix) / directory / (name + (".exe" if os.name == "nt" else ""))
    result = subprocess.run([str(executable), *map(str, arguments)],
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, (name, result.stdout, result.stderr)
    return result


def main():
    assert sys.prefix != sys.base_prefix
    prefix = Path(sys.prefix).resolve()
    for module in (tiangong, torch, np, scipy, astropy, safetensors):
        assert Path(module.__file__).resolve().is_relative_to(prefix), module.__file__
    torch.set_num_threads(2)
    torch.manual_seed(7)
    image = np.arange(32 * 32, dtype=np.float32).reshape(32, 32) - 512
    image[0, 0] = np.nan
    path = Path("input.fits")
    fits.writeto(path, image)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    data = preprocess(read_image(path))
    assert data.shape == (2, 32, 32) and np.isfinite(data).all()
    assert data[1, 0, 0] == 0 and data[0, 0, 1] < 0
    tensor = torch.from_numpy(data).unsqueeze(0)
    with torch.no_grad():
        classes = MorphologyCNN(3).eval()(tensor)
        detection = TinySetDetector(1, num_queries=3, hidden_dim=16).eval()(tensor)
    assert classes.shape == (1, 3) and torch.isfinite(classes).all()
    assert detection["boxes"].shape == (1, 3, 4)
    assert detection["logits"].shape == (1, 3, 2)
    assert torch.isfinite(detection["logits"]).all()
    assert ((detection["boxes"] >= 0) & (detection["boxes"] <= 1)).all()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    fits.writeto("cube.fits", np.zeros((2, 32, 32), dtype=np.float32))
    try:
        read_image("cube.fits")
    except ValueError:
        pass
    else:
        raise AssertionError("打包后不应静默选择科学立方体的频率平面")

    # 所有安装入口均执行真实行为，检查遗漏导入、入口错绑和相对路径问题。
    run_cli("tiangong-demo", "--output", "classification-data", "--per-class", 2)
    run_cli("tiangong-classify", "train", "--manifest", "classification-data/manifest.csv",
            "--epochs", 1, "--batch-size", 3, "--output", "classification-run")
    run_cli("tiangong-classify", "predict", "--image", "classification-data/test_0_0.fits",
            "--checkpoint", "classification-run/best.pt", "--output", "classification-predict")
    prediction = json.loads(Path("classification-predict/prediction.json").read_text())
    assert set(prediction["probabilities"]) == {"compact", "extended", "multi_component"}
    assert abs(sum(prediction["probabilities"].values()) - 1) < 1e-5
    run_cli("tiangong-demo-detection", "--output", "detection-data", "--per-split", 4, "--size", 32)
    run_cli("tiangong-detect", "train", "--manifest", "detection-data/manifest.json",
            "--size", 32, "--queries", 3, "--hidden-dim", 16, "--epochs", 1,
            "--batch-size", 2, "--output", "detection-run")
    run_cli("tiangong-detect", "evaluate", "--manifest", "detection-data/manifest.json",
            "--checkpoint", "detection-run/best.pt", "--output", "detection-evaluate")
    assert json.loads(Path("detection-evaluate/test_metrics.json").read_text())["num_samples"] == 4
    run_cli("tiangong-detect", "predict", "--image", "input.fits",
            "--checkpoint", "detection-run/best.pt", "--output", "detection-predict")
    assert Path("detection-predict/prediction.json").is_file()
    bundle_reports = {}
    for name, task, image_path, loader, model in (
        ("classification", "morphology_classification", "classification-data/test_0_0.fits",
         load_classification_checkpoint, MorphologyCNN(3)),
        ("detection", "component_detection", "input.fits", load_detection_checkpoint,
         TinySetDetector(1, num_queries=3, hidden_dim=16)),
    ):
        checkpoint = Path(f"{name}-run/best.pt")
        output = Path(f"{name}-engineering-demo.tgb")
        state = loader(checkpoint)
        model.load_state_dict(state["model"], strict=True)
        model.eval()
        run_cli("tiangong-bundle", "migrate", "--checkpoint", checkpoint, "--output", output,
                "--task", task, "--trust-checkpoint", "--source-description", "Synthetic FITS installation smoke; not scientific weights")
        inspected = json.loads(run_cli("tiangong-bundle", "inspect", "--bundle", output).stdout)
        assert inspected["config"]["release_level"] == "engineering-demo"
        # 恢复推理仅依赖bundle与新输入；原checkpoint和训练manifest不再可用。
        checkpoint.rename(checkpoint.with_suffix(".hidden"))
        manifest = next(Path(f"{name}-data").glob("manifest.*"))
        manifest.rename(manifest.with_suffix(".hidden"))
        bundle = load_bundle(output, expected_task=task, expected_classes=state["classes"])
        tensor = bundle.prepare_image(image_path)
        with torch.no_grad():
            expected, actual = model(tensor), bundle.model(tensor)
        if isinstance(expected, dict):
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        else:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        bundle_reports[name] = {"task": task, "classes": list(bundle.classes),
                                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                                "checkpoint_manifest_required": False, "prediction_exact": True}
    print(json.dumps({
        "python": platform.python_version(), "platform": platform.platform(),
        "machine": platform.machine(), "prefix": sys.prefix,
        "package_file": tiangong.__file__, "system_site_packages": False,
        "classification_shape": list(classes.shape),
        "detection_shape": list(detection["boxes"].shape),
        "versions": {name: importlib.metadata.version(name) for name in
                     ("tiangongkaiwu", "torch", "numpy", "scipy", "astropy", "safetensors")},
        "cli_workflows": ["classification train/predict", "detection train/evaluate/predict"],
        "bundle_workflows": bundle_reports,
        "data_kind": "synthetic pixels in real FITS files; not scientific validation",
    }))


if __name__ == "__main__":
    main()
