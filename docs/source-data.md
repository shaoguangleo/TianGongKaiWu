# PNG/COCO 完整射电源数据接入

本接口属于 `tiangongkaiwu 0.1.0.dev2`。它导入用户确认的完整射电源标注，输出可追溯清单与PyTorch Dataset。源级模型训练入口和源级bundle尚未接入；现有FITS分类/分量模型不能直接消费三通道PNG。

## 1. 开发与服务器环境

开发机为macOS；目标服务器为Rocky、Python3.12、NVIDIA Tesla V100。Pillow约束为 `>=12,<13`，兼容项目Python3.10—3.12；本机实际验证Pillow12.3.0。[官方Python兼容表](https://pillow.readthedocs.io/en/stable/installation/python-support.html)。

在已配置适配驱动/V100的PyTorch环境中，从项目根目录安装：

```bash
python -m pip install -e '.[dev]'
python -m pip check
python -m tiangong.source_cli --help
```

服务器的驱动和CUDA版本尚未提供，不能套用本机CPU约束。若服务器还没有PyTorch，先根据实际 `nvidia-smi` 和V100支持情况选择 [官方PyTorch安装组合](https://pytorch.org/get-started/previous-versions/)，再安装本项目。不要把 `config/constraints-source-cpu-tested.txt` 当作GPU环境锁文件。

## 2. 严格检查与派生清单

`--image-root` 应是COCO `images.file_name` 的根目录。例如file_name包含 `images/a.png` 时，根目录应为 `label-data/`，不是其下 `images/`。

```bash
python -m tiangong.source_cli prepare \
  --coco /data/HeTu/datasets/RACS-mid/label-data/result.json \
  --image-root /data/HeTu/datasets/RACS-mid/label-data \
  --output artifacts/racs-source-strict.json
```

默认严格模式遇到异常返回非零且不生成清单。原始RACS-mid导出已知包含重复身份与几何问题；它不能直接通过严格模式。显式生成清洗候选：

```bash
python -m tiangong.source_cli prepare \
  --coco /data/HeTu/datasets/RACS-mid/label-data/result.json \
  --image-root /data/HeTu/datasets/RACS-mid/label-data \
  --duplicate-policy merge_identical \
  --invalid-policy quarantine \
  --output artifacts/racs-source-candidate.json

python -m tiangong.source_cli inspect \
  --manifest artifacts/racs-source-candidate.json
```

所有输出独占创建，已有文件报错，复跑须使用新的输出文件名。原COCO/PNG不修改。`merge_identical` 只合并相同文件名且完整标注多重集相同的重复image记录；保留所有原始ID并记录changes。标注有差异、越界、无效几何或PNG等问题时，`quarantine` 隔离整张图到review，不把剩余目标或空图送入训练。默认不允许空标注图；只有确认其负样本资格后才使用 `--allow-empty`。

被隔离图需要专家复核；程序不自动裁边或重画多边形，不把不同的重复标注自动视为不同源。review-only清单可以保存；其中没有可接收图片时，Dataset和划分拒绝使用。

## 3. 标签、输入与几何契约

| 原始ID / 内部label | 类别 | 监督对象 |
|---:|---|---|
| 0 | CJ — Core Jet | 完整源 |
| 1 | CS — 致密点源 | 完整源 |
| 2 | FRI — Fanaroff–Riley I | 完整源 |
| 3 | FRII — Fanaroff–Riley II | 完整源 |

类别0是前景，不是背景。清单 `task=source_detection`、`annotation_unit=complete_radio_source`；这些标注不提供核、射电瓣等单独分量框或component→source关联真值。

PNG预处理 `png-rgb-unit-v1`：仅8-bit RGB或全不透明RGBA，去除已验证全255的alpha，RGB转为float32 CHW /255，保留原始宽高，不resize/crop。拒绝隐式透明、灰度、调色板、动画或非PNG输入。检查结构/CRC、完整zlib流及像素解码；逐次读取重新检查内容SHA256。Pillow自身 `verify()` 不执行像素解码，因此接口另执行 `load()` 和流完整性检查。[Pillow图像API](https://pillow.readthedocs.io/en/stable/reference/Image.html)。

单张PNG上限64MiB、16M像素；COCO、manifest和分组表各限64MiB普通文件，拒绝FIFO/特殊文件、重复JSON键和非有限数。只支持bbox与polygon，crowd/ignore/RLE标注整图报错或隔离。像素边界严格；polygon被bbox包围的比较仅容许 `1e-12*max(width,height,1)` 浮点误差，不修改坐标。polygon必须为简单非退化轮廓，不支持自相交或相触环；单轮廓最多1024个顶点（可重复首末闭合）。

结构定义见 [source-data-schema.json](../config/source-data-schema.json)。跨字段的ID、标签、几何、计数、合并记录、SHA256与划分关系由 `source_manifest` 再验证；JSON Schema单独通过不能代替运行验证。数据摘要支持完整性核对，不证明来源可信或科学标注正确。

## 4. 显式分组与划分

候选清单初始 `partition.status=unassigned`。旧0531划分存在已确认字节重叠，不能复用。需要按源/重叠天空/重复观测审查后提供JSON分组表：

```json
[
  {"sample_id": "image-0", "group_id": "reviewed-sky-region-A", "split": "train"},
  {"sample_id": "image-1", "group_id": "reviewed-sky-region-B", "split": "val"},
  {"sample_id": "image-2", "group_id": "reviewed-sky-region-C", "split": "test"}
]
```

以上仅格式示例，不是实际2000余张图片的划分。真实表必须完整覆盖候选清单sample IDs，不能遗漏或额外加入隔离图；同一group或相同PNG摘要不得跨区，train/val/test都须非空。不能只给每张图分配一个独立group以绕过同源审计。

```bash
python -m tiangong.source_cli split \
  --manifest artifacts/racs-source-candidate.json \
  --assignments reviewed-source-groups.json \
  --grouping-provenance '专家复核的源/天空对应表版本及依据' \
  --output artifacts/racs-source-split-v1.json
```

程序记录规范化分组映射摘要并在读取时重算，拒绝原地更改已有划分。`assigned`只表示工程映射已验证，`scientific_independence`仍为 `unverified`；仅字符串/字节检查不能证明天空独立。缺少可靠对应表时交付未分区清单和模板，不随机划出所谓独立test。

## 5. Python Dataset与服务器搬迁

```python
from torch.utils.data import DataLoader
from tiangong.source_data import SourceDataset, source_collate_fn

dataset = SourceDataset(
    "artifacts/racs-source-split-v1.json",
    "/data/HeTu/datasets/RACS-mid/label-data",
    split="train",
    # 可传 expected_sha256="冻结清单的64位小写SHA256"
)
images, targets, sample_ids = next(iter(DataLoader(
    dataset, batch_size=4, num_workers=0, collate_fn=source_collate_fn,
)))
```

返回image为 `[3,H,W]` Tensor；target包含 `boxes`（归一化cxcywh）、`pixel_boxes`（原像素xyxy）、`labels`、`polygons`、`source_ids`。空目标有规范的 `[0,4]` / `[0]` 形状；多目标/多类别完整保留。collate返回图片/target/ID列表，可容纳不同宽高，不静默padding或缩放。增加服务器workers前应验证多进程读取与吞吐。

清单只保存相对 `file_name` 和内容摘要，搬迁后显式提供服务器图片根目录。读取拒绝路径/符号链接逃逸、摘要变化与尺寸不一致。`split=None` 可用于检查未分区候选；指定train/val/test需要已划分清单。未分区数据不能作为独立评价依据。

## 6. TDD及Rocky/V100验收

在服务器项目根目录、目标Python3.12环境运行：

```bash
python scripts/validate_source_data.py \
  --suite source --require-cuda \
  --output-dir artifacts/verification-rocky-v100-source-01

python scripts/validate_source_data.py \
  --suite full --require-cuda \
  --output-dir artifacts/verification-rocky-v100-full-01
```

输出目录必须不存在。`environment.json`保存OS、Python、Pillow、PyTorch、CUDA编译版本、GPU/架构信息及带超时的 `nvidia-smi` 查询结果；`pytest.log`保存实际测试输出；`result.json`保存状态、退出码、耗时与CUDA计算结果。`--require-cuda`实际执行float32矩阵乘法、loss/backward并检查有限值；CUDA缺失/不兼容不能标通过。该检查证明基础GPU运算，不代表完整模型训练、性能或科学验收。

`source`为新增专项测试；`full`执行完整 `python -m pytest -q`，包含构建wheel/sdist并在干净环境安装的测试，需可访问依赖源或适配服务器平台的wheelhouse。可设置 `TIANGONG_TEST_WHEELHOUSE`；不能搬用macOS wheel。`TIANGONG_TEST_CONSTRAINTS`仅在已有相应服务器验证约束时设置。

本机开发可省略 `--require-cuda`；这只形成CPU开发证据。CUDA版本与驱动未知时不硬编码安装命令、不自动替换服务器依赖。官方接口参考：[PyTorch GPU架构查询](https://docs.pytorch.org/docs/stable/generated/torch.cuda.get_arch_list.html)、[NVIDIA驱动查询](https://docs.nvidia.com/deploy/nvidia-smi/index.html)。
