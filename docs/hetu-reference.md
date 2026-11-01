# HeTu 代码参考与真实标注数据接入边界

核查日期：2026-11-01。核查对象为本机 `HeTu` 与 `TianGongKaiWu` 源码；本次没有运行 HeTu 训练、下载外部数据或转换 HeTu 权重。以下路径分别相对这两个仓库根目录，行号用于定位本次检查的版本。

**结论：** HeTu 可以作为射电目标标注、类别和检测流程的参考，但其输入、模型和权重协议不能直接作为 TianGongKaiWu 的现行协议。用户已说明约有 5000 张标注图片；数量是用户提供的信息；用户已确认图片为PNG，使用Label Studio标注并导出COCO JSON。数据目录、实际数量、文件内容、类别和标注语义尚未核验。步骤03交付的是可校验的统一模型 bundle，不代表这批数据已接入或已训练出科学权重。

## 1. 已从代码确认的差异

| 项目 | HeTu 已确认行为 | TianGongKaiWu 当前行为与接入要求 |
|---|---|---|
| 主模型实现 | `train.py:21`、`train.py:24`、`train.py:30` 使用 ResNet、tensorpack 和 `tensorflow.compat.v1`；`train.py:709` 选择 ResNetFPNModel 或 ResNetC4Model | PyTorch `MorphologyCNN` 与 `TinySetDetector` 是两条工程基线，不是已移植的 HeTu、AION 或 DINO |
| 主训练输入 | `data.py:319`、`data.py:322` 以 OpenCV `IMREAD_COLOR` 读取 PNG/JPG，得到 BGR 三通道；`basemodel.py:100` 按 BGR 顺序处理均值、标准差 | `src/tiangong/data.py:12` 读 FITS 主 HDU；`src/tiangong/data.py:23` 生成 asinh 信号与有效像素掩码两个通道。PNG/JPG 不能直接交给当前 FITS 入口 |
| 尺寸与归一化 | `config.py:142` 定义短边和最大边；`config.py:147` 定义三通道均值/标准差；`data.py:297` 执行缩放和水平翻转 | 当前要求固定方形尺寸，拒绝隐式缩放或裁剪。切片级归一化不保留绝对流量，不能等同 HeTu 渲染图输入 |
| 模型恢复 | `train.py:727` 使用 `get_model_loader`；`train.py:772` 使用 `ModelSaver`。架构、类别和阈值仍通过运行配置重建 | 当前 `.tgb` 绑定受支持架构、类别顺序、预处理、权重和来源。HeTu 的 TF checkpoint 不能通过改后缀转换 |
| FITS 的用途 | `train.py:557` 仅选择 PNG；`train.py:561` 预测彩色图；`train.py:564` 随后打开同名 FITS 取得 WCS 等信息 | 这段流程不证明 HeTu 网络直接读取 FITS 科学像素。需要区分模型输入与检测后天文测量的数据来源 |

HeTu 的 `FIRST_Seg` 是另一套入口：`FIRST_Seg/train.py:15` 配置 Detectron2 Mask R-CNN，使用 `fruits_nuts` 数据注册和三类输出；`FIRST_Seg/predict.py:18` 重新构造配置并加载 `model_final.pth`。该 `.pth` 也不能被当作 TianGongKaiWu 的 checkpoint；文件扩展名不代表架构或预处理兼容。

HeTu 的 `config.py:284` 冻结并打印配置。它可以提供实验参数线索，但本次未发现上述主入口将这些参数、标签和预处理作为当前 `.tgb` 契约一起封装。参考流程时应保留信息，不应据此声称已经完成跨框架模型迁移。

## 2. 类别、框坐标和任务语义必须分别确认

HeTu `README.md:76` 给出的正式数据约定是 `data/annotations/instances_trainD1new.json`、`instances_testD1new.json`，图片置于 `trainD1new/`、`testD1new/`。`README.md:121` 的示例类别顺序为 `cs, fr1, fr2, core_jet`。这只是仓库文档中的配置，不足以确认用户现有约 5000 张图片采用相同类别或同一数据版本。

`HeTu/dataset.py:47` 从 COCO `categories` 读取类别名称，并保留原始 `category_id`；`dataset.py:204` 的检测配置在类别前加 `BG`，约定背景为 0、前景从 1 开始。**不能将原始 ID 简单减一作为通用转换规则**：类别 ID 可能不连续，类别数组顺序也可能不同。接入时应保存显式的原始 ID → 名称 → 模型索引映射，再逐项审核。TianGongKaiWu 前景索引为 `0..N-1`；检测模型背景位于最后一项，见 `src/tiangong/detection_engine.py:42`。bundle 的有序 `classes` 与 `source_category_ids` 用于保留这种对应关系；未知原始 ID 应保留未知状态。

框坐标有三个层次，不能混写：

1. HeTu 的 COCO 原标注是像素 `xywh`。`dataset.py:133` 读取后裁剪并转换成像素 `xyxy`，同时过滤部分无效或过小目标。
2. TianGongKaiWu 检测清单要求像素 `xyxy`，见 `src/tiangong/detection_data.py:89`；越界框会报错，不会静默裁剪。
3. 数据集加载器在尺寸检查后将框变成归一化 `cxcywh`，供模型和损失使用，见 `src/tiangong/detection_data.py:129`。导入器应写第 2 种格式，不能把第 3 种格式直接写入清单。

此外，标注框可能覆盖一个完整物理射电源，也可能只覆盖核、叶或其他分量。当前检测任务名是 `component_detection`，这不意味着现有数据的每个框天然具有“分量”语义。应先确认标注定义；若框覆盖完整源，需要明确相应任务协议，不能只换标签名称。一张图中若有多个不同类别的框，也不能直接导出唯一的整图分类标签；需先确定源级裁剪或整图任务规则。

HeTu `data.py:290` 的训练路径会过滤没有非 crowd 前景框的图片。TianGongKaiWu 接受 `components: []` 的负样本（`src/tiangong/detection_data.py:74`），因此不能无条件沿用此过滤规则。空场、忽略区、`iscrowd` 和未完整标注区域需要分别定义处理方式。

## 3. FITS 与标注图的像素映射需要实测

HeTu 仓库存在多种 FITS → PNG 脚本，不能仅凭文件同名假设像素坐标完全一致：

| 脚本位置 | 本次直接读取到的行为 | 接入时的影响 |
|---|---|---|
| `tools/FIRST/run_FIRST_fits2png.py:28` | DS9 Cool 配色、log、minmax 导出 | 需记录实际导出尺寸、显示设置与像素方向 |
| `tools/fits2png.py:29`、`tools/fits2png.py:51` | `zoom to fit`，按约 ±10 RMS 显示范围导出 | 需核查缩放、视口、边距及裁剪，不可只按 FITS 宽高套用框 |
| `tools/fits2png_plt_FIRST_FRI.py:42`、`tools/fits2png_plt_FIRST_FRI.py:44` | 计算了 log 变量，但实际 `imsave` 写出 `hdu.data`，使用约 ±15 RMS、`origin='lower'` | 不能依据变量名断言最终图使用 log；与 FITS 数组行索引之间的方向关系需验证 |

如果标注做在 PNG/JPG 上，而训练计划使用 FITS，应选取真实图像、对应 FITS 和真实标注做叠加核查，至少覆盖图像四角、非对称目标、边界框和多目标图。只在已确认纯垂直翻转且没有缩放、边距或裁剪时，连续边界坐标才可使用 `y1' = H - y2, y2' = H - y1`；不能预先对全量标注套用这个公式。

只有渲染图而没有原始 FITS 时，应明确建立独立的图像输入协议及对应模型训练路径。不能将三通道颜色图伪装成当前两通道科学像素输入，也不能假定仅凭渲染图能恢复原始流量或 WCS。

## 4. 仓库示例不是用户的约 5000 张射电图

本次以标准库读取了以下两个标注 JSON：

- `HeTu/demo/data/trainval.json`
- `HeTu/FIRST_Seg/data/trainval.json`

二者字节相同，SHA256 为 `3960502822ca0ced1b9737398cfca905757318a922bef9cbc5b0d2caeb444454`。每份清单记录 **18 张 800×600 图片、165 个标注**，类别为 `date`（ID 1，71 个）、`fig`（ID 2，41 个）、`hazelnut`（ID 3，53 个）。这是水果/坚果示例标注，不能据此报告已找到射电训练集或已确认 5000 张数据的标签分布。该统计来自标注 JSON，不是对每张图像内容和文件完备性的验收。

## 5. 真实数据接入的可执行步骤

| 顺序 | 要做的事 | 应形成的交付物与完成判据 |
|---|---|---|
| 1 | 确认图片/标注目录、格式、数据来源、标注对象含义和类别定义；确认是否存在配对 FITS | 数据登记说明，明确哪些信息已核验、哪些仍缺失。约 5000 张不直接等于 5000 个独立物理源 |
| 2 | 只读清点文件、图片尺寸、原始类别 ID、逐类目标数、负样本、坏图、缺图、重复样本和异常框 | 数据审计报告与异常清单；每条异常可定位，未经确认不自动丢弃 |
| 3 | 核对渲染图/FITS 的坐标、缩放和标注叠加，确认完整源框或分量框 | 有代表性的叠加样例及坐标转换测试；不同生成流程分别记录 |
| 4 | 确定任务、类别顺序和显式 ID 映射，按 TDD 实现选定格式的导入器 | 转换器、成功/失败边界测试、规范清单和来源映射。当前尚未实现这批真实数据的导入器 |
| 5 | 按物理源、重叠视野、关联天空区域、重复观测及不同分辨率版本建立组，冻结 train/val/test | 分组划分说明、清单及文件摘要、空间重叠审计。不能简单对图片逐张随机分割 |
| 6 | 在冻结数据上运行小规模训练、错误样例检查，再进行正式训练与独立测试 | 配置、日志、数据版本、评估与限制说明；测试集不用于选模。达标后再封装可共享权重 |

现有代码能够检查 `group_id` 跨分区和部分路径重复，但不能从任意组名自动证明天空区域独立，也不能凭清单 SHA256 证明图像字节没有变化。步骤 5 需要实际坐标/分组依据和图像摘要。5000 张是否足够支撑目标性能，还取决于类别分布、独立物理源数、标注一致性及目标巡天分布；本次没有给出未经实验验证的精度承诺。

**本次完成：** 代码接口核对、示例数据辨识和接入步骤定义。**待完成：** 用户真实数据定位与审计、格式转换、划分、训练和科学验收。父任务统一维护当前步骤状态与交接记录；本文件不把后续工作标记为完成。
