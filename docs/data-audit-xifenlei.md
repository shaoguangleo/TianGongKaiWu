# HeTu/xifenlei 数据接入初查

> 历史记录：用户随后纠正正式目录为 `HeTu/datasets/`。本文保留误指目录的检查结果，不作为当前训练集结论；当前数据、COCO与标注语义见 [datasets接入审计](data-audit-datasets.md)。下文“尚未找到COCO”等表述仅针对当时的xifenlei目录。

日期：2026-11-01。来源目录：`/Users/leo/share/git_internal/my/SHAO-SKA/HeTu/xifenlei/`。

**结论：目录已定位，但尚未定位到与人工标注匹配的 COCO JSON，不能开始真实监督训练。** 用户确认约有5000张PNG，使用Label Studio标注并导出COCO JSON。本次看到的是多套裁剪/渲染图、测量CSV、FITS、parquet和一个不完整读取的TAR。目录在检查期间仍有文件增加，因此以下是临时快照，不是最终样本规模或冻结数据版本。

本次只读检查HeTu：未解压归档、未删除重复图、未改图像或标注、未执行其转换/训练脚本，也没有把CSV或目录名当作人工标签。本轮只新增审计记录并更新计划/交接，步骤03的软件和已交付模型包不变。

## 1. 已看到哪些文件

完整清点快照记录4292个文件：**2454个独立PNG文件、918个FITS、899个CSV、4个parquet、10个Python脚本、6个Slurm文件和1个TAR**；未找到独立JSON。开始探查时仅有232个CSV，后续又观察到1580个CSV，证明目录尚不稳定。该快照扫描前后另有21个CSV路径状态变化。[完整统计](evidence/xifenlei-2026-11-01/inventory.json)、[逐文件记录](evidence/xifenlei-2026-11-01/files.jsonl)、[后续观察](evidence/xifenlei-2026-11-01/later-observation.json)。

“独立PNG文件”指目录中可直接读取的文件，不包括TAR成员，也不代表独立物理源。

| 子目录 | PNG数量 | 文件尺寸/颜色编码 | `label_*`目录计数 | 解释边界 |
|---|---:|---|---|---|
| `20147_final/` | 968 | 100×100，8-bit RGB | 0:324；2:81；3:563 | mask中心裁剪的一套版本 |
| `20147_maskcenter_v3/` | 968 | 462×462，8-bit RGBA | 0:324；2:81；3:563 | FITS渲染分支；抽查图含红色定位十字线 |
| `cut_parquet/` | 488 | 150×150，8-bit RGB | 0:448；2:1；3:39 | 另有918个150×150 FITS；488张PNG能按同目录结构及stem找到FITS |
| `cutout_png_mask_fast/` | 30 | 100×100，8-bit RGB | 3:30 | 快速裁剪分支在本机可直接读取的一小部分 |

表中的0/2/3只是目录中的数字，不是已确认的COCO category ID或最终科学类别。PNG签名、IHDR、chunk边界和CRC检查全部通过；没有做全量像素解码。视觉查看了3张代表图，包括同名分量的100px原PNG与462px渲染图，确认它们不能被默认视为同一像素坐标版本。

## 2. 标注文件与归档状态

在所给目录内递归查找（包括隐藏文件）没有找到独立 `.json`。TAR只读取成员列表，没有解压：

- `cutout_png_mask_fast.tar` 在读取时为502792192 bytes。
- 可读前缀列出96054个PNG成员头；其中96053个成员声明的数据范围在上述文件大小内，最后一个成员的数据范围超出文件末尾。
- 最后成员为 `cutout_png_mask_fast/20787/label_3/SB20787_component_1685a_label3_maskcenter.png`，数据起点502790656、声明长度3981、预计结束位置502794637。
- 读取报 `ReadError: unexpected end of data`；可读前缀没有JSON。**不能据此断言完整归档不含JSON，也不能把这些成员计为已完整校验的可训练图片。**

目录仍在变化，归档读取不完整可能与尚未完成的传输有关，本次未判定其是否永久损坏。应先确认复制/同步完成，再重新校验完整性和清点。

需要用户提供 **Label Studio导出的COCO JSON确切路径**。拿到JSON后，才可逐项验证 `images.file_name` 指向哪套PNG、原始 `category_id` 与名称、每图标注数、空标注图、bbox边界、crowd/ignore和缺失文件；目前这些事项均未完成。

## 3. 重复与类别目录重叠

- 2454个PNG文件只有2444个不同的文件SHA256：发现10组相同字节的文件，共10份额外副本。未删除任何文件。[重复明细](evidence/xifenlei-2026-11-01/duplicate-png.json)。
- `20147_final`与`20147_maskcenter_v3`各有968张图，但各自只包含924个不同的文件名component token；两套版本的924个token集合一致。仅按字节去重不能识别这些不同渲染版本。[配对统计](evidence/xifenlei-2026-11-01/pairing.json)。
- 在这两个目录中，各有43个component token出现在多个 `label_*` 目录，共产生44份额外路径。例如 `SB20147_component_1080a` 同时出现在label_0和label_2。**这提示标签/来源需要核对，不足以直接判定人工标注冲突。** [全部重叠记录](evidence/xifenlei-2026-11-01/component-label-overlap.json)。

文件名token、唯一字节数量都不能替代独立物理源数量。不同component也可能属于同一物理源或重叠天空区域；同源不同裁剪、渲染和观测版本不能随机分入train与test。

## 4. 从生成代码确认的接入风险

以下代码位置相对 `HeTu/`；只审阅实现，没有执行。所用小型来源脚本的SHA256保存在 [reference-sources.json](evidence/xifenlei-2026-11-01/reference-sources.json)。

| 事项 | 代码事实 | 对训练接入的要求 |
|---|---|---|
| 标签含义 | `xifenlei/cut_parquet/cut_unique_catalog_maskcenter_fast.py:65`声明0:CJ、1:CS、2:FRI、3:FRII，但该常量未参与实际处理；`mask_measure_fast.slurm:44`只选择0/2/3 | 不能用该常量或目录名替代COCO categories，不能将原始ID简单减一 |
| 输入与裁剪 | `batch_mask_measure_cutouts_fast_no_preview.py:159`配对上游PNG、150px header FITS和CSV；`:366`读取RGB，`:367`读取RLE mask，`:393`按mask中心裁剪及补零为100px | 先确定人工标注针对150px原图还是100px派生图，保留裁剪原点、补边与方向关系 |
| 框坐标 | 同文件`:435`复制原CSV行，只新增测量项，没有把原150px的bbox/counts转换到100px输出坐标 | 不能直接把catalog CSV的bbox当作当前PNG的监督框；尺寸缩放也不等于裁剪坐标变换 |
| CSV性质 | 抽查CSV含score、bbox、RLE counts、天球中心、流量/RMS/SNR，代码从上游检测目录读取 | 更像检测/测量产物，这是来源推断；不视为已验证的人工COCO真值 |
| 检查渲染图 | `cutout_5.py:478`创建绘图，`:486-489`加入红色十字线并savefig；`:193`还存在按坐标选择flip的流程 | 462px检查图不应默认混入原图训练；需核对是否被人工标注及实际变换，不能仅按同名配对 |
| parquet分支 | `cut_unique_catalog_maskcenter_fast.py:814`按绝对银河纬度5°区分normal与galactic_b5；`:866`优先用unique_source_id命名 | 这不是train/val/test划分；`test_maskcenter_3`目录名也不构成科学盲测身份 |
| PNG显示 | 同脚本`:440`处理非有限值，`:447`裁掉负值，`:463`用大FITS的RMS归一化，`:499`用viridis、origin=lower渲染 | RGB PNG不能还原为原始流量矩阵；建立独立、有版本的PNG预处理契约 |
| FITS处理 | 同脚本`:366-413`填充NaN，`:916-926`保存实际cut_data及WCS header | 文件含像素，不是仅header；但原始无效mask和完整观测元数据不一定保留 |

918个FITS的头卡扫描均显示二维150×150、BITPIX=-32、RA/Dec类型字段；均未在所读header中看到BUNIT/BMAJ/BMIN。另抽查两份95040-byte文件，包含实际数据区。未进行全量像素有限性、WCS坐标一致性、流量或beam科学验收，不能据此当作原始科学FITS直接复用。

Slurm脚本引用的原始数据位于集群 `/groups/hetu_ai/...` 路径。本机没有其所指的 `HeTu/racsmid/test/png` 和 `HeTu/xzj_code/fits` 目录，不能仅凭脚本路径声称原始大FITS和制作配置已经齐备。

## 5. 后续动作与完成判据

1. 确认目录复制完成，补充COCO JSON路径，重复清点并冻结来源清单及摘要。
2. 用COCO的file_name逐图定位准确PNG版本；统计缺图、重复图、类别、bbox边界、每图目标数与负样本，不自动丢弃异常。
3. 确认框表示完整物理源还是分量；核对43个多label token与不同渲染版本，建立可追溯的人工标签映射。
4. 选择少量真实图和框做叠加核对，确认坐标、裁剪/翻转、RGBA通道、尺寸及颜色处理；禁止自动把PNG伪装为FITS。
5. 按TDD实现PNG输入适配和COCO导入器，并把预处理新版本绑定模型包。当前v1 FITS bundle保持原契约。
6. 按物理源、重叠天空与重复观测分组，冻结train/val/test后再开始真实训练；缺少天空信息时明确独立性证据不足。

本次完成的是目录定位、快照清点、来源代码核对及问题清单。没有确认“约5000张独立标注图”、没有验证COCO标注框、没有导入/划分训练集，也没有启动训练。步骤07—10继续推进，不能标为科学数据验收完成。
