# Python 包安装、构建与调用

发行名为 `tiangongkaiwu`，导入名为 `tiangong`，当前版本 `0.1.0.dev1` 是本地工程预览。尚未上传 PyPI，以下安装均使用本仓库或本地构建产物；包中不含科学权重。

实现集中在 `src/tiangong/`。根目录11个同名 Python 文件是兼容入口；须先安装，之后原来的 `python main.py ...`、`python detect.py ...` 等命令仍可用。兼容入口不修改 `sys.path`，也不会将 `data`、`model` 等通用名称安装为全局模块。

## 1. 环境与依赖边界

- 元数据允许 Python `>=3.10,<3.13`。实际验证的解释器、平台和结果见 [打包验证](package-verification.md)；允许安装范围不等于所有平台均已验收。
- 运行依赖：NumPy、Astropy、PyTorch、SciPy；`dev` extra 另含 pytest、build 和构建后端。
- NumPy 限定 `<2`，避免既有 PyTorch 2.2.2 与 NumPy 2 的数组互操作问题。放宽范围需要新的兼容性验证。
- [CPU 直接依赖约束](../config/constraints-bundle-cpu-tested.txt) 仅用于 Python 3.10/macOS x86_64 复测，不是跨平台完整锁文件。原生 arm64、Linux、Windows、其他 Python 版本和 CUDA 须分别验证。
- 不在本步决定公开分发许可证或托管方式，也不将工程包称为正式科学模型 release。

## 2. 开发安装与旧入口

在仓库根目录运行；**新建 `.venv` 时必须显式指定 Python 3.10 及以上（且 <3.13）的解释器**，例如 `python3.10`、`python3.11` 或 `python3.12`，不要依赖未加版本号的 `python`/`python3`：

```bash
python3.10 -m venv .venv   # 或 python3.11 / python3.12
source .venv/bin/activate
python -c "import sys; assert sys.version_info >= (3, 10) and sys.version_info < (3, 13), sys.version"
python -m pip install -e ".[dev]"
python -m pip check
python -m pytest -q
```

已有 `pip install -r requirements.txt` 用法仍支持，但须在仓库根目录执行；该文件转发到 `.[dev]`，依赖版本只在 `pyproject.toml` 维护。可在安装命令中加 `-c config/constraints-bundle-cpu-tested.txt` 复测上述 CPU 组合。

完整测试包含从实际 wheel/sdist 安装到独立虚拟环境的测试，首次运行需要获取依赖、额外磁盘空间和时间。测试不会继承 `system-site-packages`，也不以 editable 安装代替用户产物验收。

## 3. 构建并从产物安装

在已安装开发 extra 的环境中，从仓库根目录构建：

```bash
python -m build
```

产物为 `dist/tiangongkaiwu-0.1.0.dev1-py3-none-any.whl` 和 `dist/tiangongkaiwu-0.1.0.dev1.tar.gz`。构建默认先生成 sdist，再从 sdist 构建 wheel；测试另从这两种产物分别安装。wheel 的 `py3-none-any` 仅说明本项目代码为纯 Python，不意味着 PyTorch 等依赖在所有平台都有可用产物。

在仓库外使用另一全新虚拟环境安装 wheel；将下例路径替换为实际构建产物路径：

```bash
python3.10 -m venv /tmp/tiangong-user-venv
/tmp/tiangong-user-venv/bin/python -m pip install /path/to/tiangongkaiwu-0.1.0.dev1-py3-none-any.whl
/tmp/tiangong-user-venv/bin/python -m pip check
/tmp/tiangong-user-venv/bin/tiangong-classify --help
/tmp/tiangong-user-venv/bin/tiangong-detect --help
```

将安装路径换成 `.tar.gz` 可验证 sdist。不要同时用两种产物覆盖同一验收环境，避免掩盖缺少文件的错误。源码包包含测试、旧入口、安装说明和 CPU 约束；安装运行时仅使用 `tiangong` 包及其声明的依赖。

## 4. 安装后的入口与 Python 调用

| 已安装命令 | 对应原入口 | 行为 |
|---|---|---|
| `tiangong-classify` | `main.py` | 分类 train / evaluate / predict |
| `tiangong-detect` | `detect.py` | 分量检测 train / evaluate / predict |
| `tiangong-demo` | `demo.py` | 分类工程模拟 FITS |
| `tiangong-demo-detection` | `demo_detection.py` | 检测工程模拟 FITS 与框标注 |
| `tiangong-bundle` | `python -m tiangong.bundle` | checkpoint迁移及完整模型包校验/查看 |

参数保留原行为；也可使用 `python -m tiangong.main`、`python -m tiangong.detect` 等模块入口。这些是已有任务的安装入口，尚非步骤04的统一推理 SDK。

例如在仓库外、激活安装环境后运行模拟分类：

```bash
tiangong-demo --output example-data --per-class 2
tiangong-classify train --manifest example-data/manifest.csv --epochs 1 --output example-run
tiangong-classify predict --image example-data/test_0_0.fits --checkpoint example-run/best.pt --output example-predict
```

用新输出目录运行，保留现有数据保护规则。这里的训练和权重只证明工程流程，不构成真实射电科学验收。

包内 Python 调用示例（输入须为已明确选面的二维 FITS 切片）：

```python
import torch
from tiangong.data import read_image, preprocess
from tiangong.model import MorphologyCNN

image = read_image("example-data/test_0_0.fits")
tensor = torch.from_numpy(preprocess(image)).unsqueeze(0)
model = MorphologyCNN(num_classes=3).eval()  # 随机初始化；不是科学预训练模型
with torch.no_grad():
    logits = model(tensor)
print(logits.shape)
```

检测模型可从 `tiangong.detection_model` 导入。保留原 `state_dict` 键；步骤03补齐了分类 checkpoint 契约并新增统一 bundle。新命令 `tiangong-bundle migrate/inspect` 及 Python 导出/加载详见 [模型包说明](bundle.md)。校准、embedding、微调与批量 SDK 仍属于后续步骤。

## 5. 离线重复安装测试

预先准备包含运行和构建依赖的、与目标 Python/平台匹配的 wheelhouse。设置以下变量时，安装测试只从该目录解析依赖：

```bash
TIANGONG_TEST_WHEELHOUSE=/path/to/wheelhouse \
TIANGONG_TEST_CONSTRAINTS=/path/to/constraints-cpu-tested.txt \
python -m pytest -q tests/test_package_install.py
```

两变量均可省略；省略时 pip 按常规方式获取依赖。约束文件只适用于其声明的平台组合。测试复制样例脚本至仓库外，以 `python -I` 执行，并检查项目及五个运行依赖均来自新环境；FITS 样本由测试生成，像素仍是模拟数据。

打包方式已核对 [PyPA 打包教程](https://packaging.python.org/en/latest/tutorials/packaging-projects/)、[src 布局说明](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/) 和 [setuptools 配置文档](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html)。

步骤03增加运行依赖 `safetensors>=0.8,<1`，实际验证0.8.0；新 `constraints-bundle-cpu-tested.txt` 引用并保留步骤02的 `constraints-cpu-tested.txt`，增加safetensors精确版本；当前安装环境的完整freeze另见模型包验证证据。发布bundle使用无pickle格式；旧 `.pt` 入口只用于可信本地实验文件，限制详见 [模型包安全边界](bundle.md#4-文件契约和完整性)。
