# HeTu/datasets：PNG/COCO 源级标注接入审计

日期：2026-11-01。只读来源：`/Users/leo/share/git_internal/my/SHAO-SKA/HeTu/datasets/`。

**结论：已定位 Label Studio 的 RACS-mid COCO 导出及全部对应 PNG，并由用户确认每个框/多边形表示完整射电源。当前候选为2007条图片记录、2003张不同图片、2499条标注；尚未清洗、划分冻结或训练，不能当作已验收的约5000张独立训练图。**

用户已纠正先前误指的 `HeTu/xifenlei/`；[旧目录初查](data-audit-xifenlei.md)仅保留历史记录，其中的复制、TAR和裁剪问题不适用于本报告中的数据集。本轮没有修改HeTu文件、解压归档、运行转换或训练程序。

## 1. 可直接定位的文件与规模

以下路径相对 `HeTu/datasets/`；解析 `images.file_name` 时须使用表中的图片根目录，不能使用JSON所在的 `annotations/` 目录。

| 导出 | COCO JSON | 图片根目录 | 图片记录 / 不同文件 | 标注记录 |
|---|---|---|---:|---:|
| 新完整导出 | `RACS-mid/label-data/result.json` | `RACS-mid/label-data/` | 2007 / 2003 | 2499 |
| 旧train | `RACS-mid/classified-data/annotations/Astronomy_instances0531_train_1606.json` | `RACS-mid/classified-data/` | 1606 / 1603 | 1926 |
| 旧val | `RACS-mid/classified-data/annotations/Astronomy_instances0531_val_401.json` | `RACS-mid/classified-data/` | 401 / 401 | 476 |

三份JSON均声明 `contributor=Label Studio`。新导出的 `date_created` 为2024-06-06，旧train/val为2024-05-31；这是导出元数据，不是已经锁定的巡天release。三份JSON中的PNG引用全部可解析，均为150×150、8-bit RGBA；JSON宽高与PNG头一致。不同文件数也与各表内不同PNG字节摘要数一致，但不代表不同物理源数。

整个 `datasets/` 快照包含7886个文件，其中7649个PNG、138个FITS、3个直接JSON和13个ZIP。这里混有RACS/FIRST等数据与重复版本，不能把PNG总数算作这套标注集规模。仅查看13个ZIP的直接成员目录，未见JSON成员；没有递归解压嵌套ZIP，不能据此断言归档内不存在其他标注。

完整统计、JSON摘要和验证范围见 [inventory.json](evidence/datasets-2026-11-01/inventory.json)；逐图片路径、尺寸与SHA256见 [新导出](evidence/datasets-2026-11-01/label_studio_0606-resolved-images.jsonl)、[旧train](evidence/datasets-2026-11-01/train_0531-resolved-images.jsonl)、[旧val](evidence/datasets-2026-11-01/val_0531-resolved-images.jsonl)。

## 2. 已确认的标注语义与类别

用户明确确认：**每个框/多边形标注完整射电源，不是核、喷流或射电瓣等单独分量**。用户随后更正术语：FRI/FRII按 **Fanaroff–Riley射电形态分类**，先前称“Seyfert（塞佛特）”是名称写错。这项确认应进入后续标签契约；尚未完成逐例专家复核或类间边界裁决。

| COCO原始ID | 标签 | 已确认含义 | 新导出标注数 | 旧train | 旧val |
|---:|---|---|---:|---:|---:|
| 0 | CJ | Core Jet，核—喷流形态 | 382 | 313 | 61 |
| 1 | CS | 致密点源；对应观测分辨率下的紧致形态 | 1660 | 1254 | 318 |
| 2 | FRI | Fanaroff–Riley I | 73 | 54 | 18 |
| 3 | FRII | Fanaroff–Riley II | 384 | 305 | 79 |
| 合计 | | | 2499 | 1926 | 476 |

FR分类依据射电亮度结构；Seyfert I/II按光谱区分，二者不是同一标签体系。[Fanaroff & Riley原始论文](https://academic.oup.com/mnras/article/167/1/31P/2604898)、[NASA/ESA关于Seyfert光谱分类的说明](https://science.nasa.gov/missions/hubble/hubble-sights-a-galaxy-with-forbidden-light/)。本文不据名称为任何单例补充宿主、光谱或物理类型证据。

这些数量按annotation记录计，重复记录尚未消除。CS占多数，FRI仅73条，后续应报告逐类有效源数与不确定性，不能只看总体accuracy。原始 `category_id=0` 是前景CJ，不是背景；导入器必须保存原始ID与内部连续索引的显式映射，不能统一减一。

## 3. 图片与几何标注检查

新导出中每图1/2/3/4个目标的图片记录分别为1581/363/60/3；没有空标注记录。**228张不同图片包含多个类别**，因此不能给整张图强行分配单一类别。可用完整源框开展源级检测；若开展单源分类，需先定义裁剪、上下文及邻源处理规则。

全部2499条标注为多边形，`iscrowd=0`、`ignore=0`；image/annotation ID唯一，外键引用有效。面积为有限正数，polygon鞋带公式面积与记录值在1e-6容差内一致。检查发现以下问题：

| 问题 | 具体证据 | 处理要求 |
|---|---|---|
| 同一图片多条image记录 | 3个文件名分别被3/2/2个image ID引用，额外4条记录 | 合并身份前聚合全部标注；记录旧ID映射和复核决定 |
| 左边界轻微越界 | ann2345、image1879，`images/0ff18a07-SB20333_component_2414a.png`；bbox x=-0.16475216269992146 | 对应polygon也越界；不得仅改bbox或静默接受 |
| 上边界轻微越界 | ann2357、image1889，`images/2fa47e3a-SB20333_component_2446a.png`；bbox y=-0.16919555519045965 | 需明确边界修正规则、重算相关几何并保留原件和变换日志 |
| 图片根目录有2张未引用PNG | `images/ebe2175f-cj.cs_SB41554_component_756a.png`、`images/b47d5f63-cj.cs_SB41554_component_1389a.png` | 它们出现在旧split中；不能自动当作负样本 |

重复图片的完整路径及标注关系如下，详见 [duplicate-image-details.json](evidence/datasets-2026-11-01/duplicate-image-details.json)：

| 文件名（均位于images/） | image ID | annotation记录数 / 不同标注内容数 |
|---|---|---:|
| `1e0579a4-cs_SB41554_component_1248a.png` | 639、640、641 | 6 / 2 |
| `1473c320-SB20333_component_704a.png` | 1582、1583 | 2 / 1 |
| `e02e43f0-SB20333_component_2301a.png` | 1844、1845 | 3 / 3 |

“不同标注内容”忽略 `id/image_id` 后比较其余字段。最后一组有3份不同内容，不能简单保留第一个image记录并丢弃其余标注；也不能未经专家确认就把3份都当作不同物理源。详细异常、多类别图和额外PNG见 [annotation-issues.json](evidence/datasets-2026-11-01/annotation-issues.json)。

旧train/val未发现越界框、多边形边界或外键问题；旧train有2条空标注image记录，其负样本资格未确认。各split根目录的“额外PNG”按该split单独计数，通常包括另一split引用的图片，不代表目录整体未标注。

## 4. 新旧版本和旧划分泄漏

新导出与旧train+val合并都含2007条image记录、2003个不同文件名，但图像集合不相同：新版本独有 `6be9e825-SB20333_component_1072a.png`、`9b95f873-SB20333_component_1078a.png`，旧版本独有第3节列出的两张未引用PNG。两目录共有2001个被JSON引用的文件名，其PNG字节均相同。

按 `file_name` 聚合所有重复image ID的标注，再忽略 `id/image_id` 比较标注多重集：2000个共同有标注文件名中，**213个的标注内容不同**。这包括几何、类别或记录数量等任一字段差异，不等于213个类别发生变化。旧版本共2402条annotation，新版本2499条；不能把新标注套用旧split后仍称同一冻结版本。

**确认有1张相同图片跨旧train/val：** `images/1e0579a4-cs_SB41554_component_1248a.png`，train中出现2条记录、val中1条，SHA256为 `73f5a342e9c0387b60bdb8183cfaa97ce667cf290044da97a0f198537bd1cd2b`。这是直接内容泄漏，旧划分不能原样作为本项目独立评价依据。

共有79个 `SB<id>_component_<number>` 父前缀、2个SBID同时出现在train/val，79中包含上述已知重叠对应的 `SB41554_component_1248`。这些前缀是需要核对的同源/重叠候选，不能凭文件名断言79组都是同一物理源或实际天空泄漏。后续需要原始坐标、图像足迹与来源关系。详见 [versions-and-splits.json](evidence/datasets-2026-11-01/versions-and-splits.json)。

## 5. HeTu代码可证实的协议

只读审查 `HeTu/pjlab/configs/_astronomy_/astronomy_data2k_mask-rcnn_hrnetv2p-w32_1024_lr1e-4_iter20k.py`：

- 第59行类别顺序为CJ/CS/FRI/FRII，第101、115行引用上述0531 train/val JSON；第57行使用集群路径。
- 第9行BGR→RGB，第74行载入bbox与mask，第75—77行执行Resize/RandomCrop/RandomFlip。这是三通道图像流程，不能直接等同TGKW的FITS信号/mask双通道。
- 第120行 `test_dataloader = val_dataloader`，第129行 `test_evaluator = val_evaluator`；此配置没有独立test。

配置文件只证明所写协议，不证明训练已经执行或得到某种性能。没有找到生成这两份0531划分的明确脚本、随机种子或天空分组记录；仓库其他2023-12-21项目的随机划分脚本不能作为其来源。Label Studio启动文档也不提供本导出的科学标注准则。[参考源码摘要](evidence/datasets-2026-11-01/reference-sources.json)。

## 6. 后续实施顺序与验收

| 顺序 | 在做什么 | 完成后应得到什么 | 当前状态 |
|---|---|---|---|
| 07.1 | 定位COCO并解析准确图片版本 | 路径、原始字节摘要、类别/几何问题清单 | 已完成本轮只读检查 |
| 08.1 | 确认监督对象与类别含义 | 完整源；CJ/CS/FRI/FRII含义和原始ID | 用户已确认；逐例专家验收未完成 |
| 07.2/08.2 | 固定选用导出版本，处理重复身份、标注差异及越界 | 原件不变、可追踪的派生清单、清洗规则与待复核表 | 待实现/复核 |
| 09 | 按源/观测/天空关系重新分区，锁定独立测试 | 无已知字节泄漏，未知关联显式记录；冻结协议与清单 | 未完成 |
| 10 | TDD实现PNG/COCO导入和源级任务契约 | RGB/RGBA策略、框/多边形一致变换、显式标签映射、受控失败 | 待开发 |
| 11/12 | 训练源级检测与按源裁剪分类基线 | 逐类指标、错例、适用范围与可恢复权重 | 未启动 |

后续TDD用例至少覆盖：原始类别0、缺图、重复image身份及标注合并冲突、路径越界、越界/退化几何、空图、多类别图、RGBA策略、变换后框/掩膜一致、跨split同内容和可追溯导出。PNG解码与归一化需单独版本化，不能改扩展名伪装成FITS。

应明确新增**完整源检测**语义，不得把本数据静默装入当前 `component_detection` 契约。现有集合匹配、框回归可作为实现参考；已有分量检测bundle保持原义。分量框与component→source关系尚缺，不能用完整源框证明分量检测或关联能力。

## 7. 检查边界与复核记录

本轮执行JSON结构/引用/有限数值/边界/面积检查、PNG签名/头/chunk边界/CRC及SHA256核对，检查期间来源文件大小与mtime未变化。PNG检查**不包含全部图像像素解码**；仅视觉查看两张代表图，没有完成全量标注叠加或专家语义验收。数据许可、release、WCS/beam/RMS及原始FITS映射尚未冻结；不从PNG颜色恢复物理流量。

本轮只更新审计、计划与handoff；没有新增运行代码，因此不重复模型测试，也不重新构建步骤03产物。最终检查见 [verification.json](evidence/datasets-2026-11-01/verification.json)，本轮证据摘要见 [SHA256SUMS](evidence/datasets-2026-11-01/SHA256SUMS)。步骤03的315 passed/2 CUDA skipped是既有工程验收，不能解释为本数据已通过训练或科学评价。
