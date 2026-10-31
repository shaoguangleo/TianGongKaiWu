# 最小检测器验证记录（2026-10-31）

结论：分量标注 → FITS 张量 → 集合预测 → Hungarian 匹配 → CE/L1/GIoU → checkpoint → 独立 test → 单图预测的工程链路已完成。没有真实 RACS 或 GPU 科学性能证据，模拟测试的检测质量也尚不足以验收模型性能。

## 当前环境与边界

- Python 3.10.9；macOS 26.6.2 x86_64；CPU，PyTorch 线程数 2。
- NumPy 1.26.4、torch 2.2.2、SciPy 1.10.0、Astropy 6.1.7、pytest 9.2.0。
- 执行器：`/tmp/tiangong-detection-venv/bin/python`。该临时 venv 通过 system-site-packages 继承当前已有依赖，仅在 venv 中安装兼容的 NumPy；**不是全新环境的依赖解析验收**。全局环境未降级。
- 原有系统 NumPy 2.2.6 与 torch 2.2.2 的 `torch.from_numpy` 实测失败，不能把“旧 5 项测试通过”当作数据链路可用。
- 实测直接依赖约束见 [constraints](../config/constraints-cpu-tested.txt)，详细平台、源码 SHA256、checkpoint SHA256、清单 SHA256 见 [environment.json](evidence/detection-2026-10-31/environment.json)。工作区改动尚未提交；记录包含开发前 HEAD。

历史 `VERIFICATION.md` 保留原样，其环境与分类实验结果不代表本次检测实现。

## 自动测试

```bash
# tests/：最终源码的完整验证命令
PYTHONDONTWRITEBYTECODE=1 /tmp/tiangong-detection-venv/bin/python -m pytest -q \
  -o cache_dir=/tmp/tiangong-detection-final-cache
```

结果：**123 passed，1 skipped，34.63s**。唯一跳过项是无 CUDA 时无法运行的 device 匹配测试。原 5 项分类测试包含在总数中。

覆盖真实 FITS→Tensor、空图与变长目标、连续像素边界、坏标注/越界/跨分区身份、手算 IoU/GIoU、匹配排列不变、背景监督、两头梯度、两张不同图过拟合、固定阈值一对一统计、重复框 FP、评估参数不变、真实 CLI 保存/恢复/评估/预测、异常参数、坏 checkpoint、结果防覆盖及原子保存失败保护。

保留证据：[完整 pytest 输出](evidence/detection-2026-10-31/pytest.txt)。`git diff --check` 与新增文本文件的 whitespace 检查无错误。

TDD 过程：数据新测试先观察 25 项缺失功能/极值溢出失败，再实现；模型/损失和运行层先观察新模块缺失失败，再实现；生成器先观察 23 项缺失失败，再实现。最终审查修复的 5 项回归测试先全部失败，再全部通过，见 [RED](evidence/detection-2026-10-31/review-red.txt) / [GREEN](evidence/detection-2026-10-31/review-green.txt)。

## 端到端 CPU 实验

最终源码执行了下列命令，所有命令退出码为 0。数据与 checkpoint 位于临时目录；可持久化审查的清单、配置、日志、训练历史与预测 JSON 已复制到本仓库的 evidence 目录。

```bash
# demo_detection.py：12 train / 12 val / 12 test，包含空图和1–3分量图
PYTHONDONTWRITEBYTECODE=1 /tmp/tiangong-detection-venv/bin/python demo_detection.py \
  --output /tmp/tiangong-detection-smoke.SnXYK8/data --per-split 12 --size 64 --seed 7

# detect.py：20轮训练，按val loss保存最佳模型
PYTHONDONTWRITEBYTECODE=1 /tmp/tiangong-detection-venv/bin/python detect.py train \
  --manifest /tmp/tiangong-detection-smoke.SnXYK8/data/manifest.json \
  --output /tmp/tiangong-detection-smoke.SnXYK8/train-final \
  --size 64 --queries 6 --hidden-dim 32 --epochs 20 --batch-size 4 --seed 42

# detect.py：使用冻结清单的独立test，不调整阈值
PYTHONDONTWRITEBYTECODE=1 /tmp/tiangong-detection-venv/bin/python detect.py evaluate \
  --manifest /tmp/tiangong-detection-smoke.SnXYK8/data/manifest.json \
  --checkpoint /tmp/tiangong-detection-smoke.SnXYK8/train-final/best.pt \
  --output /tmp/tiangong-detection-smoke.SnXYK8/evaluate-final \
  --batch-size 4 --score-threshold 0.5 --iou-threshold 0.5

# detect.py：单图推理
PYTHONDONTWRITEBYTECODE=1 /tmp/tiangong-detection-venv/bin/python detect.py predict \
  --image /tmp/tiangong-detection-smoke.SnXYK8/data/test_0001.fits \
  --checkpoint /tmp/tiangong-detection-smoke.SnXYK8/train-final/best.pt \
  --output /tmp/tiangong-detection-smoke.SnXYK8/predict-final --score-threshold 0.5
```

| 观测项 | 结果 |
|---|---:|
| 训练 loss，第1轮→第20轮 | 8.266091 → 2.589998 |
| 最佳 val loss / 轮次 | 3.595194 / 第20轮 |
| test 图像数 / 真值分量数 | 12 / 18 |
| TP / FP / FN | 6 / 30 / 12 |
| precision / recall / F1 | 0.166667 / 0.333333 / 0.222222 |
| 成功匹配平均 IoU | 0.668425 |

这些数值是固定 score=0.5、IoU=0.5 的模拟检测统计，**不是 mAP，也不是科学可靠度**。误报仍多，不能把小样本过拟合与 loss 下降解释成实际检测质量达标。没有根据 test 指标继续调参或做真实数据性能声明。

证据：[训练日志](evidence/detection-2026-10-31/train.log)、[配置](evidence/detection-2026-10-31/config.json)、[历史](evidence/detection-2026-10-31/history.json)、[测试指标](evidence/detection-2026-10-31/test_metrics.json)、[逐图预测](evidence/detection-2026-10-31/test_predictions.json)、[单图预测](evidence/detection-2026-10-31/prediction.json)、[生成标注](evidence/detection-2026-10-31/manifest.json)。临时数据清理后需使用新目录重新生成，不能直接复用本页旧输出目录。

## 独立审查与修复

独立代理只读审查未发现 Critical，确认两项 Important，已统一修复：

1. **清单变更导致假独立测试**：evaluate 原本只检查当前清单内部隔离，交换 train/test 后仍可运行。现在与 checkpoint 清单 SHA256 比较，校验和解析同一份字节；更改 split/标签/路径/格式会拒绝。外部新清单评估不在本版范围。
2. **checkpoint 直接写入可能跟随 symlink 或截断旧 best**：改为独占临时文件完整保存，首次独占发布、后续原子替换；测试证明保存抛异常后旧文件字节不变、临时文件清理、symlink 目标不被覆盖、启动后出现的同名文件不被覆盖。

审查还指出分区读取测试缺口：新增真实 CLI 回归，训练时让 test FITS 不存在；训练后恢复 test、移除 train/val FITS，清单不变仍能评估，证明非当前分区图像不进入当前计算。无遗留审查问题。

## 尚未验证

真实 RACS 标注/天空隔离、FITS 内容版本、波束尺度定位、AP/mAP、科学选择效应、物理分量关联、CUDA/MPS、跨硬件一致性、网络文件系统的发布语义均未验证。当前产物适合继续开发与受控实验，不是生产或科学发布验收。
