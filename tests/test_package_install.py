"""验证真实分发产物，防止漏打模块、依赖和 CLI，或借用仓库完成导入。"""
from pathlib import Path
import importlib
import json
import os
import posixpath
import re
import shutil
import subprocess
import sys
import tarfile
import venv


ROOT = Path(__file__).resolve().parents[1]


def _run(arguments, cwd, log):
    """子进程不继承源码导入路径；保存完整诊断以便复现安装失败。"""
    environment = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME"):
        environment.pop(name, None)
    environment["PYTHONNOUSERSITE"] = "1"
    result = subprocess.run(
        list(map(str, arguments)), cwd=cwd, env=environment,
        capture_output=True, text=True, timeout=600,
    )
    log.write_text(result.stdout + result.stderr, encoding="utf-8")
    assert result.returncode == 0, log.read_text(encoding="utf-8")
    return result.stdout


def test_legacy_imports_share_packaged_implementation():
    # 同一模块对象保证旧调用者对 detect 私有函数的 monkeypatch 仍作用于实现。
    for name in (
        "data", "model", "engine", "main", "detect", "demo", "demo_detection",
        "detection_data", "detection_model", "detection_losses", "detection_engine",
    ):
        legacy = importlib.import_module(name)
        try:
            packaged = importlib.import_module(f"tiangong.{name}")
        except ModuleNotFoundError as error:
            raise AssertionError("尚无可安装的 tiangong 包") from error
        assert legacy is packaged, f"{name} 存在两套实现，兼容调用可能漂移"


def test_wheel_and_sdist_run_in_clean_environments(tmp_path):
    # 从副本构建，避免测试产生的 egg-info/build 修改工作区。
    source = tmp_path / "source"
    shutil.copytree(
        ROOT, source,
        ignore=shutil.ignore_patterns(
            ".git", ".venv*", "__pycache__", ".pytest_cache", "*.egg-info",
            "build", "dist", "*.fits", "*.pt", ".vscode",
        ),
    )
    distributions = tmp_path / "dist"
    _run(
        [sys.executable, "-I", "-m", "build", "--no-isolation", "--outdir", distributions, source],
        tmp_path, tmp_path / "build.log",
    )
    wheels = list(distributions.glob("*.whl"))
    sdists = list(distributions.glob("*.tar.gz"))
    assert len(wheels) == len(sdists) == 1, "必须交付 wheel 与可独立重建的 sdist"

    # 源码包必须包含安装/验证说明实际引用的本地资料，不能只在原仓库可读。
    with tarfile.open(sdists[0]) as archive:
        members = {member.name for member in archive.getmembers() if member.isfile()}
        for name in sorted(members):
            if not name.endswith(".md"):
                continue
            document = archive.extractfile(name).read().decode("utf-8")
            for link in re.findall(r"\[[^\]]+\]\(([^)]+)\)", document):
                if "://" in link or link.startswith("#"):
                    continue
                target = posixpath.normpath(posixpath.join(posixpath.dirname(name), link.split("#")[0]))
                assert target in members, f"源码包缺少 {name} 引用的 {link}"

    # 可选 wheelhouse 用于断网复测；默认允许 pip 正常解析项目声明的依赖。
    pip_options = ["--disable-pip-version-check", "--no-input"]
    if wheelhouse := os.environ.get("TIANGONG_TEST_WHEELHOUSE"):
        pip_options += ["--no-index", "--find-links", str(Path(wheelhouse).resolve())]
    if constraints := os.environ.get("TIANGONG_TEST_CONSTRAINTS"):
        pip_options += ["--constraint", str(Path(constraints).resolve())]

    reports = {}
    for kind, artifact in (("wheel", wheels[0]), ("sdist", sdists[0])):
        environment = tmp_path / f"{kind}-venv"
        venv.EnvBuilder(with_pip=True, system_site_packages=False).create(environment)
        assert "include-system-site-packages = false" in (environment / "pyvenv.cfg").read_text()
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        outside = tmp_path / f"{kind}-user"
        outside.mkdir()
        # pip 对 sdist 使用独立构建环境，不能复用 source 副本或现有 wheel。
        _run([python, "-I", "-m", "pip", "install", *pip_options, artifact],
             outside, tmp_path / f"{kind}-install.log")
        _run([python, "-I", "-m", "pip", "check"], outside, tmp_path / f"{kind}-check.log")
        script = outside / "package_smoke.py"
        shutil.copyfile(ROOT / "tests/package_smoke.py", script)
        output = _run([python, "-I", script], outside, tmp_path / f"{kind}-smoke.log")
        reports[kind] = json.loads(output)
        assert reports[kind]["classification_shape"] == [1, 3]
        assert reports[kind]["detection_shape"] == [1, 3, 4]
        assert set(reports[kind]["bundle_workflows"]) == {"classification", "detection"}
        assert reports[kind]['source_data_workflow'] == {
            'shape':[3,4,6], 'category_zero_preserved':True,
            'relocated_root':True, 'assigned_splits':['train','val','test'],
        }
        _run([python, "-I", "-m", "pip", "freeze"], outside, tmp_path / f"{kind}-freeze.txt")
    (tmp_path / "installation-report.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
