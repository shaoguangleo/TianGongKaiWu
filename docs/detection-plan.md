# 最小分量检测器实施方案

用户于 2026-10-31 确认下一步为“最小检测器：分量标注、匹配与框回归”。

## 范围与选择

沿用 PyTorch 与二维、固定尺寸 FITS 输入。采用 CNN + 二维位置编码 + Transformer decoder + 固定查询的单尺度集合检测器；优先验证可变数量目标、背景查询和一对一匹配，比先实现多尺度 DINO 更适合当前小型基线。模型从随机权重训练，不下载权重，不接入 AION，不把分量框视为物理源或科学测量。

标注使用独立 JSON，不复用分类 CSV：`schema_version: 1`、`coordinate_system: "pixel_xyxy"`、`samples`。每项包含 `sample_id/path/split/group_id/components`，每个分量包含 `component_id/label/box`。坐标是 NumPy 图像列 x、行 y 的连续像素边界，左上为 (0,0)，右下可等于 (W,H)，框必须满足 x1<x2、y1<y2；不是 FITS/WCS 一基坐标。空 components 是有效负样本。默认唯一前景类 `radio_component`；形态标签留在既有源级分类任务。

## 文件与接口

- `detection_data.py`：`read_detection_manifest(path, num_classes) -> list[dict]` 验证全清单后选择 split；拒绝空白 ID、重复 ID、跨分区 group/path、坏标签、非有限/倒置/越界框。
- `DetectionDataset(rows, size)` 返回 `(image, target)`：image 为 `[2,H,W]`；target 包含 `sample_id/path/component_ids/labels/boxes`；labels 是 `[N]` int64，boxes 是 `[N,4]` float32、归一化 cxcywh。`detection_collate(batch)` 返回 `(stacked_images, targets)`。
- `detection_model.py`：`TinySetDetector(num_classes, num_queries=10, hidden_dim=64)` 返回 `{"logits": [B,Q,C+1], "boxes": [B,Q,4]}`，最后一类为 no-object，框为归一化 cxcywh。
- `detection_losses.py`：`match_predictions(outputs, targets)` 返回逐图预测/真值索引对；代价为 `-P(class)+5*L1+2*(1-GIoU)`。`detection_loss(outputs, targets)` 返回 `loss/loss_ce/loss_bbox/loss_giou`；未匹配查询监督为背景（权重 0.1），总损失为 CE+5L1+2GIoU，框损失按目标总数归一化。空图参与背景训练，目标数超过查询数明确报错。
- `detection_engine.py`：推理转换成像素 xyxy；按分数降序、同类别、IoU 阈值的一对一评估，重复框计 FP。报告 precision/recall/F1、TP/FP/FN、匹配平均 IoU，不称为 mAP。`run_detection_epoch` 训练/评估共用，不在评估阶段更新参数。
- `detect.py`：独立 `train/evaluate/predict` CLI。训练只使用 train/val，依据 val loss 选优；checkpoint 记录 schema、预处理版本、类别顺序、结构、清单 SHA256、随机种子；evaluate 强制匹配冻结清单字节且仅访问 test。拒绝覆盖已有实验目录，checkpoint 完整临时写入后独占发布/原子替换。
- `demo_detection.py`：生成含 0–3 个高斯分量的模拟 FITS 与框标注；独立 split/group、拒绝覆盖。合成框仅用于工程验证。
- `tests/test_detection_*.py`：数据约束、匹配排列不变、空目标、框损失梯度、过拟合、重复预测、CLI 完整路径与坏参数。

## TDD 与验收

- [x] 数据层：先运行新测试观察缺失功能失败，再实现；补充真实 FITS→Tensor 路径及共享预处理极值回归测试。
- [x] 模型/损失：先测试解析匹配、空目标和框损失，再实现小模型；验证分类与框头均获梯度及小样本可过拟合。
- [x] 运行层：先测试重复预测/空图指标与 CLI，再实现训练、恢复、独立测试、单图预测和模拟数据。
- [x] 全套 pytest、临时目录端到端训练/评估/预测、独立代码审查、`git diff --check`，记录当前真实环境和结果。结果：123 passed、1 CUDA skipped；详见 `docs/detection-verification.md`。

保留已有 `runs/`、`demo_data/` 和分类入口；不自动提交、推送或安装到全局 Python。当前系统 PyTorch 2.2.2 与 NumPy 2.2.6 的 Tensor 桥接已确认不兼容，验证使用临时隔离环境，不把历史 VERIFICATION.md 的环境当作当前环境。

参考已核对的官方接口：[SciPy assignment](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html)、[PyTorch TransformerDecoderLayer](https://docs.pytorch.org/docs/2.15/generated/torch.nn.TransformerDecoderLayer.html)、[NumPy ABI 排障](https://numpy.org/doc/stable/user/troubleshooting-importerror.html)。实际兼容性以本次运行验证为准。

## 审查决策记录

- 采用小型单尺度集合检测而非完整 DINO；目的是先验证标注、匹配与框梯度，代价是没有多尺度、去噪及预训练能力。
- 依用户指定目录就地新增独立检测文件，保留现有未跟踪数据，不创建分支或自动提交。
- 默认 evaluate 锁定完整清单 SHA256，关闭重新划分训练样本后冒充独立测试的路径；代价是清单格式变化和外部新数据集均不能直接复用该评估入口，需要后续显式协议。
- 独立审查两项 Important（清单未冻结、直接保存 best 跟随 symlink/可能截断旧 checkpoint）以及分区读取测试缺口，统一通过回归测试修复；不扩展到物理源关联或科学性能验收。
