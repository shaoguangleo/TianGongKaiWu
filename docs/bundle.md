# 统一模型包 v1：导出、迁移与加载

步骤03将现有两种模型的权重、架构、类别顺序和预处理绑定到一个 `.tgb` 文件。工程软件版本为 `0.1.0.dev1`；schema/architecture 版本均为1。安装见 [installation.md](installation.md)，实现过程见 [bundle-plan.md](bundle-plan.md)，验证见 [bundle-verification.md](bundle-verification.md)。

## 1. 支持什么

| task | 架构 | 输入 | 原始模型输出 |
|---|---|---|---|
| `morphology_classification` | `MorphologyCNN` | 固定尺寸二维主HDU FITS，2通道 | `[B,C]` logits，类别按 labels 顺序 |
| `component_detection` | `TinySetDetector` | 同上 | logits `[B,Q,C+1]`，背景为最后一类；boxes `[B,Q,4]` 为归一化 `cx,cy,w,h` |

输入先按 `radio-asinh-mask-v1-f64` 预处理：有限像素估计背景和尺度、asinh信号变换与有效掩码，输出float32。尺寸必须精确匹配，拒绝自动缩放、真正多平面立方体及全无效图。允许8—4096像素；这是契约上限，不是内存/性能保证。

分类2—1024类；检测1—1024前景类，queries为1—1024，hidden_dim为4—512且能被4整除。当前为原始网络输出，阈值、框到天球坐标、批量结果 schema 和科学校准仍需后续实现。

## 2. 迁移已有本项目 checkpoint

先按安装说明安装项目，在任意目录运行；目标必须是新文件。`--trust-checkpoint` 明确确认旧 `.pt` 来自你自己的可信实验；缺少该参数时，在读取/反序列化前拒绝。不要对下载的未知文件使用该开关：

```bash
tiangong-bundle migrate \
  --checkpoint /path/to/classification/best.pt \
  --task morphology_classification --trust-checkpoint \
  --output /path/to/models/classification-v1.tgb \
  --source-description '本项目分类实验；填写真实数据和实验来源'

tiangong-bundle migrate \
  --checkpoint /path/to/detection/best.pt \
  --task component_detection --trust-checkpoint \
  --output /path/to/models/detection-v1.tgb \
  --source-description '本项目检测实验；填写真实数据和实验来源'

tiangong-bundle inspect --bundle /path/to/models/classification-v1.tgb
```

新版分类 checkpoint 写入 task/schema/preprocess/architecture；检测 checkpoint 已有对应契约。旧分类文件没有预处理版本时，默认拒绝迁移，只有确认它确实来自相同预处理实现后，才在上述分类命令中增加：

```bash
--legacy-preprocess-version radio-asinh-mask-v1-f64
```

`tiangong-classify evaluate/predict` 加载旧分类 checkpoint 也需要该参数。迁移从同一份文件字节快照计算 SHA256 并加载，记录原始摘要、格式与人工确认状态。缺失的代码 revision、完整训练配置、数据 release 和图像字节摘要保留为缺失，不能从文件名补造。迁移产物统一为 `engineering-demo`，不会因可以加载而自动获得科学发布资格。

仅支持本项目已知格式。HeTu 的 TensorFlow/tensorpack 或 Detectron2 `.pth` 权重不能仅改后缀迁移。数据输入和标签差异见 [HeTu参考审计](hetu-reference.md)。

## 3. Python调用

以下代码在项目安装后运行，bundle 和输入路径须替换为本机文件：

```python
import torch
from tiangong.bundle import load_bundle

bundle = load_bundle(
    "/path/to/models/classification-v1.tgb",
    expected_task="morphology_classification",
)
image = bundle.prepare_image("/path/to/new-input.fits")
with torch.inference_mode():
    scores = bundle.model(image).softmax(dim=-1)[0].tolist()
print(dict(zip(bundle.classes, scores)))
```

加载器返回 CPU/eval 模型；输入预处理也在 CPU。`expected_classes=[...]` 可以额外约束确切类别顺序；不传时以模型包自身有序 labels 为准。不要用默认三类名字替换包内标签。分数未经校准。

训练后可直接导出受支持的原始模型，无须另存 checkpoint：

```python
from tiangong.bundle import export_bundle

export_bundle(
    model,  # 已训练的、未改造结构的 MorphologyCNN
    "/path/to/models/new-classifier.tgb",
    task="morphology_classification",
    classes=["FR I", "FR II"],  # 必须与训练时整数标签顺序完全一致
    size=64,
    source_category_ids=[7, 42],  # 可省略；未知时保存为null，不能推断为0/1
    provenance={"source": "填写真实实验来源", "manifest_sha256": manifest_sha256},
    release_level="engineering-demo",
)
```

`model` 和 `manifest_sha256` 来自调用者真实训练结果。导出不改变原模型的 train/eval 状态。只接受注册架构及float32有限稠密权重；改换 activation、attention/decoder 配置、额外 hooks 或结构必须先定义新架构版本，不能伪装成v1。`research` 是允许的来源级别标签，不代表已通过科学验收；`science` 不被接受。

## 4. 文件契约和完整性

`.tgb` 是具有固定六个成员的 ZIP，不解压到文件系统：

| 成员 | 内容 |
|---|---|
| `config.json` | schema、task、architecture/parameters、软件版本、release_level、required_assets |
| `labels.json` | 有序classes；可选且同序的source_category_ids |
| `preprocessing.json` | 版本、输入格式、尺寸、通道、dtype及拒绝resize |
| `provenance.json` | 来源、训练清单摘要、seed/epoch、迁移原件摘要、缺失信息 |
| `weights.safetensors` | CPU state_dict；保留的 `__tiangong_contract_sha256__` uint8张量保存32字节契约摘要 |
| `checksums.json` | 以上五个文件的SHA256 |

[bundle-schema.json](../config/bundle-schema.json) 描述四份metadata合并后的 JSON Schema（Draft2020-12），其中 `$defs.checksums` 描述摘要对象。运行时进一步检查任务/架构匹配、category ID长度、严格整数类型、权重keys/shape/dtype/有限值、ZIP与契约摘要。JSON Schema不代替加载器，也不用于执行外部引用。

单包及累计解压数据最多256MiB，每个JSON最多1MiB；成员必须恰好匹配白名单，拒绝重复路径、额外文件、路径穿越、符号链接、FIFO等非普通文件、加密及不支持的压缩。v1仅支持无需额外资产的两个架构，`required_assets` 非空即拒绝，不能漏掉codec/tokenizer仍然宣称可用。

写入先在目标同目录完成临时文件并fsync，再以hard link独占发布；已存在目标、符号链接和竞争写入均不覆盖。文件系统不支持该原子发布方式会明确失败，不回退到覆盖旧文件。更新模型需使用新文件名。

发布bundle使用 safetensors 的 bytes API，加载过程不调用 `torch.load`、不执行pickle、不动态导入架构。恢复后为CPU模型，严格校验权重及保留契约摘要。SHA256可发现损坏和不一致，不能认证来源或防止攻击者重写整包；仍应核实来源。预处理、代码本身及依赖环境也属于复现条件。

旧 `.pt` checkpoint 是单独的兼容边界：现有分类/检测入口及显式迁移仍用 `torch.load(weights_only=True)`，只能读取可信的本地实验输出。当前验证的PyTorch2.2.2受已知反序列化漏洞影响；`weights_only=True` 不是对恶意文件的安全保证。官方公告记录了 [CVE-2025-32434](https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6)（≤2.5.1）与 [CVE-2026-24747](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p)（≤2.9.1，修复版本≥2.10.0）。本轮通过改用非pickle发布格式消除bundle对该加载路径的依赖，未宣称修复旧PyTorch库或完成新PyTorch版本的平台验收。

safetensors实际测试版本为0.8.0，接口核对 [官方Torch API](https://huggingface.co/docs/safetensors/api/torch)。实现也核对了 [PyTorch序列化说明](https://docs.pytorch.org/docs/stable/notes/serialization.html)、[Python ZIP接口](https://docs.python.org/3.10/library/zipfile.html) 与 [JSON Schema Draft2020-12](https://json-schema.org/draft/2020-12)。本轮实际运行版本以验证记录为准，不等同于最新文档所列版本均已验证。

## 5. 下一步与真实数据

步骤04将提供统一单图/批量SDK与CLI，并定义可追溯结果。当前 `load_bundle` 与 `prepare_image` 是恢复契约所需接口，不宣称已完成该完整用户流程。

用户已说明有约5000张标注图片，并确认使用Label Studio标注PNG、导出COCO JSON；实际目录、类别表与框语义尚待核验。当前FITS预处理不能直接读取这批PNG，需要单独定义PNG输入契约及COCO导入器，不能通过改后缀或伪装FITS接入。步骤07—10将先检查样例、标注对象语义、FITS或渲染坐标映射、重复与天空分组，再决定转换和训练。当前两份示例 bundle 只来自模拟FITS工程测试，不能用作真实射电科学权重。
