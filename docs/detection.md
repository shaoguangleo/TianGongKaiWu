# 最小射电分量检测器

这是独立于 `main.py` 分类基线的可训练检测实现。输入是包含零个或多个辐射分量的二维 FITS；输出是分量类别、模型分数和像素框。默认只有 `radio_component` 一个前景类别。框之间的物理源关联、源级形态分类、WCS 测量、多尺度 DINO 和 AION 均在后续阶段。

## 运行

代码语法要求 Python 3.10 或以上。先创建独立环境，安装 PyTorch 对应 CPU/CUDA 版本和 `requirements.txt`。本次 CPU 实测的直接依赖保存在 `config/constraints-cpu-tested.txt`，并非跨平台锁文件；复现实测组合请使用 Python 3.10，较新解释器不能直接照搬该约束文件。

```bash
# config/constraints-cpu-tested.txt：使用本次 CPU 验证的依赖组合
python -m venv .venv-detection
source .venv-detection/bin/activate
python -m pip install -r requirements.txt -c config/constraints-cpu-tested.txt
```

当前机器已有 Python 的 torch 2.2.2 与 NumPy 2.2.6 组合无法执行 `torch.from_numpy`；本次在 `/tmp` 创建独立环境安装 NumPy 1.26.4 后验证通过。不要直接降级其他项目共用的全局 Python。相关背景见 [NumPy 官方 ABI 排障](https://numpy.org/doc/stable/user/troubleshooting-importerror.html)。

以下命令各使用一个新目录，重复运行须更换输出目录：

```bash
# demo_detection.py：生成模拟 FITS 和分量框，绝非真实 RACS 科学样本
python demo_detection.py --output demo_detection_data --per-split 12 --size 64 --seed 7

# detect.py：train 只读取 train/val 图像，按验证损失选择 best.pt
python detect.py train --manifest demo_detection_data/manifest.json \
  --output detection_runs/experiment_001 --size 64 --queries 10 \
  --hidden-dim 64 --epochs 20 --batch-size 4 --seed 42

# detect.py：evaluate 只读取 test 图像，阈值须事先在验证集确定
python detect.py evaluate --manifest demo_detection_data/manifest.json \
  --checkpoint detection_runs/experiment_001/best.pt \
  --output detection_runs/evaluation_001 --batch-size 4 \
  --score-threshold 0.5 --iou-threshold 0.5

# detect.py：单图预测从 checkpoint 恢复类别、尺寸和模型结构
python detect.py predict --image demo_detection_data/test_0001.fits \
  --checkpoint detection_runs/experiment_001/best.pt \
  --output detection_runs/prediction_001 --score-threshold 0.5

# tests/：包括真实 FITS、训练/恢复/独立评估/预测的集成测试
python -m pytest -q
```

默认 CPU；其他设备通过 `--device cuda` 或 `--device mps` 请求。设备是否存在会先检查，但本次只验证 CPU，不能据此宣称 CUDA/MPS 已可用。模型没有外部预训练权重或自定义 CUDA 算子。

## 标注格式

```json
{
  "schema_version": 1,
  "coordinate_system": "pixel_xyxy",
  "samples": [
    {
      "sample_id": "tile01_cutout01",
      "path": "fits/tile01_cutout01.fits",
      "split": "train",
      "group_id": "sky_region_01",
      "components": [
        {
          "component_id": "tile01_cutout01_c0",
          "label": 0,
          "box": [10.0, 12.0, 28.0, 30.0]
        }
      ]
    }
  ]
}
```

将示例保存为自己的 `manifest.json`；它只展示结构，运行训练还须提供独立 `val` 分区，运行评估须有 `test` 分区。

- `path` 相对于清单目录解析，支持绝对路径。`sample_id` 全清单唯一，同一解析后的路径不能出现两次。
- `group_id` 必须由上游依据天空区域、重叠和重复观测定义。同组不跨分区；程序拒绝首尾空白标识，防止无意绕过检查。仅此检查不能证明真实天空隔离。
- `box=[x1,y1,x2,y2]` 是连续像素边界；x 是 NumPy 列方向，y 是行方向，左上边界为 `(0,0)`。64×64 图的右下边界是 `(64,64)`，像素中心是 `(column+0.5,row+0.5)`。这不是 FITS/WCS 一基坐标，不对 y 轴做显示翻转。
- 框必须有限、在图像内并有正面积；`label` 是从 0 开始的前景类别索引。使用多类时，在 train 传 `--classes class_a class_b` 并固定标签定义；最后的 no-object 类由训练器内部处理，不能写入标注。
- `components=[]` 是有效空背景图，不可把空图丢弃。单图目标数不得超过 `--queries`，超出会报错，不能静默丢弃目标。
- 所有图像必须与训练的 `--size` 相同，不会自动裁剪或缩放。真实立方体须先依据科学任务选取平面。预处理保留负值和独立有效区域掩码；全无效图会报错。

## 实现与损失

`detection_data.py` 复用 `data.py` 的 FITS 和 asinh+mask 预处理。统计使用 float64 避免 float32 极值归约溢出，最终张量为 float32。Dataset 把像素 xyxy 转成归一化 cxcywh，空目标保留 `[0,4]` 形状。

`detection_model.py` 实现三层 stride-2 CNN、二维正弦位置编码、两层四头 Transformer decoder、可学习查询、类别头和框头。每个查询输出 `C+1` 个 logits（最后是背景）与 sigmoid cxcywh。结构较小且采用单尺度，不等同于 DINO，也未引入去噪、多尺度或检测预训练。

`detection_losses.py` 使用 [SciPy linear_sum_assignment](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html) 解离散匹配：

`cost = -P(真值类别) + 5 × L1(cxcywh) + 2 × (1 − GIoU)`。

匹配自身不反传，匹配后的框及类别损失反传。未匹配查询学习背景，CE 背景权重为 0.1；框损失按整个 batch 的目标总数归一化。总损失为 `CE + 5 × L1 + 2 × GIoU_loss`。纯空图 batch 的框损失为保留计算图的零，仍可训练背景。

模型代码使用的 decoder 参数已对照 [PyTorch 官方接口](https://docs.pytorch.org/docs/2.15/generated/torch.nn.TransformerDecoderLayer.html) 核对；当前实测版本仍以验证记录为准。

## 输出与评价

| 命令 | 制品 |
|---|---|
| train | `best.pt`、`config.json`、`history.json` |
| evaluate | `test_metrics.json`、`test_predictions.json` |
| predict | `prediction.json` |

checkpoint 保存结构、类别顺序、schema、预处理版本、随机种子、训练轮次及清单 SHA256；恢复必须兼容这些契约，使用 `weights_only=True`，不会回退到任意 pickle 加载。仍应只加载可信 checkpoint。清单哈希不是 FITS 文件内容校验和，尚不具备完整科学数据版本追溯。

`evaluate` 必须使用训练时冻结清单的相同字节版本，包括原来的 train/val/test 划分；修改标签、split、路径、顺序甚至 JSON 格式均会因 SHA256 不一致而拒绝。校验与解析使用同一份字节，防止把训练图重新标成 test 后报告独立结果。本版不提供外部清单评估模式；复制整个数据目录时相对路径清单无需改写。train 不读取 test FITS，evaluate 不读取 train/val FITS，这两点通过令非当前分区文件不存在的真实 CLI 测试验证。

最佳 checkpoint 先完整写入同目录独占临时文件并 flush/fsync，首次发布独占目标名，后续更新原子替换；保存失败不会截断已有 best，目的 symlink 会被拒绝。输出文件系统须支持同目录 hard link 与原子替换，本次仅在本地文件系统验证。

推理以包含背景的 softmax 为分数；若最高类别为背景则丢弃，否则按阈值保留。归一化框转换成像素 xyxy、裁到图像内并丢弃退化框。不做 NMS，重复预测会在评价中计为 FP。

评价按分数降序，将同类别且 IoU 达标的预测与未命中的真值一对一配对，报告 TP/FP/FN、precision、recall、F1、逐类 support 及成功匹配的平均 IoU。没有匹配时平均 IoU 定义为 0。**这是固定分数/IoU 阈值下的统计，不是 AP/mAP**，也没有按波束归一化定位误差。阈值不应通过 test 调优。

## 当前边界

模拟数据是独立噪声加网格附近的高斯，框取生成中心的正负 3σ。它只支持流程检查、匹配验证和小样本过拟合；不模拟 ASKAP 波束、相关噪声、旁瓣、截断大源、复杂源混叠及科学标注分歧。成功运行和模拟定位不能证明真实 RACS 检测性能。

当前未实现断点续训、分布式训练、全巡天切片合并、物理分量关联、AP/mAP、旋转增强、WCS/波束测量和科学选择效应验证。大图、真实标注与科学验收需要后续独立工作。
