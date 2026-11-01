# TianGongKaiWu 当前交接：PNG/COCO源级数据接入已实现

日期：2026-11-01。当前开发版：`0.1.0.dev2`。数据步骤07—10的接入工程已实现；科学划分、Rocky/V100验收及源级模型训练未完成。步骤03的既有bundle保持可用，软件步骤04统一推理SDK仍待开发。

## 本轮编码交付

用户确认macOS用于开发，实际测试服务器为Rocky、Python3.12、NVIDIA Tesla V100；驱动/CUDA版本尚未给出。遵循TDD实现，只在证据支持的范围内报告本机结果，未登录或运行服务器测试。

| 小步 | 在做什么 | 已完成的制品与边界 |
|---|---|---|
| 07/08 数据导入 | 校验COCO、PNG、几何与重复标注 | `coco_import.py`、`png.py`、`source_geometry.py`；原件只读，原ID与摘要保留，默认严格，显式merge_identical/quarantine |
| 09 清单与分组 | 绑定标签、预处理、来源与划分 | `source_manifest.py`、`config/source-data-schema.json`；组/内容不跨区、分组摘要重算、防覆盖，科学独立性unverified |
| 10 数据读取 | 服务器换根目录仍可按同字节读取 | `source_data.py`；三通道RGB、归一化框＋像素框/多边形、空目标/多目标、batch列表、逐次摘要检查 |
| 操作与验收 | 可执行CLI、测试和GPU环境记录 | `source_cli.py`、`scripts/validate_source_data.py`；六个专项测试文件，source/full模式，CUDA实际matmul/backward，结果落盘 |

真实派生制品：`dist/source-data/racs-mid-20240606-candidate-v1.json` 含 **1999张PNG、2484条标注**，逐类CJ/CS/FRI/FRII为376/1653/72/383。合并2组完全相同的重复image记录；隔离4张图：image38的ann83轮廓自相交/相触、image1844/1845重复身份下标注不同、image1879和1889含越界框。隔离图的所有目标一起进入review，不形成部分漏标的训练图。详情见 [实际导入记录](evidence/source-data-2026-11-01/real-import.json)。

分组模板为 `dist/source-data/racs-mid-20240606-groups-template.json`，group/split均null，必须依据可靠源/天空对应表填写，不能直接用于训练或独立评价。用户明确表示暂时没有对应表，授权本轮交付未分区候选清单和模板，当前清单unassigned。用户原约5000张完整数据范围尚未核验。

用法见 [源级数据接入](source-data.md)，设计/TDD步骤见 [实施计划](source-data-plan.md)，测试、环境问题和安装证据见 [验证记录](source-data-verification.md)。Pillow新增运行依赖 `>=12,<13`；本机实际12.3.0。`config/constraints-source-cpu-tested.txt`仅供macOS CPU开发复现，不可作为Rocky/V100约束。

独立审查已修复PNG截断zlib仍被Pillow接受、自相交轮廓、分组JSON特殊文件/重复键、分组摘要漂移、清洗policy与记录不一致等问题，均有RED/GREEN证据。首次full因本机磁盘满出现1失败/47错误，516通过/2CUDA跳过；日志保留为 `pytest-full-disk-error.txt`。只清理本任务前序安装测试的可重建临时venv，保留所有源码、原数据、日志和dist产物，之后复验。最终完整回归 **564 passed、2 CUDA skipped（435.15s）**；wheel/sdist干净安装和实际源级流程均通过。dev2两份软件产物、候选与模板的摘要在 `dist/SOURCE-DATA-SHA256SUMS`，服务器验收尚未运行。

下一数据步骤：服务器运行 `python scripts/validate_source_data.py --suite full --require-cuda --output-dir <新的结果目录>`；专家复核隔离图并补齐源/天空映射，再生成真实划分，接入源级训练及独立bundle契约。当前仅Dataset与数据管线就绪，旧FITS双通道模型不直接接收PNG，也没有新训练权重。本轮未git add/reset/commit。 收尾发现暂存区被外部更新，已保留当前索引，没有回退；文档最终补录部分仍在工作区。

## 前轮记录：步骤07只读审计与08标注语义确认

用户将正式目录纠正为 `HeTu/datasets/`。本轮只读清点并形成 [datasets数据审计](data-audit-datasets.md) 与 `docs/evidence/datasets-2026-11-01/` 原始记录；先前xifenlei报告仅保留误指目录的历史检查，其问题不套用于本数据。没有改图、修改标注、解压、转换或训练。

- 实际COCO：`HeTu/datasets/RACS-mid/label-data/result.json`；图片根目录为同级 `label-data/`，`file_name`含 `images/`。2007条图片记录引用2003张不同PNG，2499条标注；图片全部存在，均150×150 RGBA。目录另有2张未被当前JSON引用的PNG。
- 用户确认每个框/多边形表示**完整射电源**，CJ=Core Jet、CS=致密点源，并更正FRI/FRII为**Fanaroff–Riley射电形态**，此前“Seyfert”是名称写错。原始ID为0/1/2/3，标注数382/1660/73/384；保留ID映射，0不是背景。
- 发现3组同文件不同image ID、2条框与polygon轻微越界、228张多类别图。标注记录数不能直接解释为独立源数；整图不能直接作为单标签分类样本。
- 旧0531 train/val位于 `RACS-mid/classified-data/annotations/`，与新导出标注版本不同：2000个共同有标注文件中213个标注多重集不同。旧train/val有1个字节完全相同的PNG跨分区；共有79个父component前缀候选（含已知重叠对应前缀）需坐标核验，不能自动认定同源。对应HeTu配置复用val作为test，没有独立盲测。
- 已完成JSON引用、几何、PNG结构/CRC与摘要检查和两张代表图视觉查看；未做全量像素解码、全量叠加图、专家语义验收或天空隔离。所见2003张候选图不代表约5000张数据已全部定位与验收。
- 本轮仅更新审计、计划与交接。源码及既有wheel/sdist/bundle沿用步骤03版本，未重跑模型测试或重建包；本轮文档、数据摘要与工作区保护复核见 [verification.json](evidence/datasets-2026-11-01/verification.json)。未执行git add/reset/commit。

下一数据动作：固定选用的标注版本，保留原件并明确重复/越界修正规则，复核重复记录中的不同标注，建立源/天空分组与新划分，再以TDD实现PNG/COCO输入和**完整源检测**契约。现有 `component_detection` 仍保持分量语义，本数据不能替代分量与关联真值。步骤07尚未完成冻结，软件步骤04仍为下一主线编码工作。

## 本次目标与结论

依据 [统一20步计划](PLAN.md)，在步骤02可安装包基础上完成统一模型bundle，并参考HeTu核对后续真实数据接口。本轮已在纠正后的 `HeTu/datasets/` 定位RACS-mid COCO与PNG、确认完整源和四类语义并留存审计；尚未完成真实训练集清洗、独立划分、冻结或训练。

## 实际完成的代码与制品

| 小步 | 做什么 | 已实现的结果 |
|---|---|---|
| 03.1 契约 | 绑定模型与其解释方式 | `config/bundle-schema.json`；task、架构版本、类别顺序/原始ID、预处理、来源、发布级别与摘要 |
| 03.2 恢复 | 导出并受控重建现有双模型 | `src/tiangong/registry.py`、`bundle.py`；六成员`.tgb`，白名单架构、严格权重校验、CPU/eval恢复、输入尺寸检查 |
| 03.3 分类保护 | 与既有检测入口保持同等级保障 | `classification_checkpoint.py`、`main.py`、`data.py`；新版checkpoint、原子保存、防覆盖、冻结清单字节校验；旧预处理显式确认 |
| 03.4 迁移与安装 | 从可信原件迁移并在仓库外使用 | `tiangong-bundle migrate/inspect`；原件单次字节快照与SHA256；实际wheel/sdist安装及两个任务的训练/恢复流程 |
| 03.5 HeTu与数据 | 确认可参考项及不可直接套用项 | [HeTu参考审计](hetu-reference.md)；类别/背景索引、输入通道、框坐标、完整源/分量语义、图像渲染映射和数据接入步骤 |

软件版本为 `0.1.0.dev1`，步骤02的dev0产物保留。新增运行依赖safetensors0.8.0；`config/constraints-bundle-cpu-tested.txt`引用原CPU约束，原文件不变。完整用法与接口见 [bundle.md](bundle.md)，安装见 [installation.md](installation.md)。根目录 `handoff.md` 仍是本文导航。

## 设计决定与修复

- bundle成员是config、labels、preprocessing、provenance四个JSON，加 `weights.safetensors` 与 `checksums.json`。权重中保存32字节元数据摘要，避免只换类别顺序并重算外层摘要后被静默接受。
- 仅注册原始MorphologyCNN和TinySetDetector。除权重外还核对模块实例配置，修复Transformer `norm_first`、activation、attention布局/heads不出现在state_dict或extra_repr而导致恢复语义变化的问题。
- 最多256MiB，单JSON最多1MiB；不解压路径、不动态导入架构；拒绝缺成员、重复成员、未知/缺失资产、额外路径、符号链接与非普通文件。以非阻塞打开后fstat拒绝FIFO，避免在检查前等待writer。
- 临时完整文件在同目录fsync后通过hard link独占发布，保留已有文件和竞争写入结果。修改版本需新目标名。
- 官方安全核对发现当前PyTorch2.2.2有已知weights_only反序列化漏洞，因此最终发布格式切换为safetensors；bundle加载完全不调用torch.load。旧`.pt`迁移要求 `trusted_checkpoint=True` / `--trust-checkpoint`，未确认则在读取前拒绝。详见 [安全边界及官方公告](bundle.md#4-文件契约和完整性)。
- 未知代码revision、数据release、原始图像摘要和完整训练配置保留缺失状态。HeTu TensorFlow/Detectron2权重不自动转换；5000张数据不默认映射为RACS-mid或现有三类标签。

## TDD与验收证据

- 新接口先RED：bundle23项、分类checkpoint64项；工厂和CLI另有RED记录。之后补测结构语义、FIFO、非安全反序列化路径和显式信任门槛。
- 初版全套：310 passed、2 CUDA skipped（451.54s）；安全核对后代码变化，因此不以此作为最终格式验收。
- 最终专项：**190 passed、1 CUDA skipped（10.17s）**，包括禁用torch.load后的双模型往返、伪装pickle拒绝、旧格式迁移、防覆盖、工厂边界与分类契约。
- 最终完整回归 **315 passed、2 CUDA skipped（448.96s）**；最终归档代码摘要与测试产物一致，wheel/sdist再次安装和运行通过。两份交付bundle及FITS已从安装环境产出并重新加载验证。日志、实际环境、详细命令见 [bundle-verification.md](bundle-verification.md)。安装测试使用无system-site-packages的新环境，仓库外 `python -I` 运行FITS、双模型、五个CLI；移走checkpoint和训练manifest后恢复bundle，要求预测逐位一致。
- 每次测试样本均为真实FITS格式内的模拟像素，不是用户真实数据或科学验收。本助手未执行git add/reset/commit。收尾时发现暂存区在任务期间被更新，已保留当前索引且未回退；最终状态与PNG/COCO说明的部分修订仍未暂存。源码、配置、MANIFEST和新增文档需要由用户一并审核后手工提交。

## 交付路径

交付产物位于 `dist/tiangongkaiwu-0.1.0.dev1-py3-none-any.whl`、`dist/tiangongkaiwu-0.1.0.dev1.tar.gz`；两份工程bundle与配套FITS位于 `dist/bundles/`，摘要清单位于 `dist/STEP03-SHA256SUMS`。该目录由Git忽略，不自动提交。最终安装复验和摘要记录另存 `docs/evidence/bundle-2026-11-01/final-artifacts.json`，在归档后形成，不回写已校验归档。

## 适用边界与下一步

本轮实际验证目标是macOS兼容环境的x86_64 Python3.10.9、PyTorch2.2.2、NumPy1.26.4、SciPy1.10.0、Astropy6.1.7、safetensors0.8.0、CPU；不代表原生arm64、CUDA、Linux/Windows或其他版本组合已验收。首次加载原生扩展在沙箱中停滞，测试采用已批准的沙箱外执行，依赖离线安装。

现有分类/检测`.pt`入口及迁移只能处理可信本地实验输出，不能因为设置weights_only而处理来源未知的文件。bundle采用非pickle格式解决发布加载路径问题，不代表旧PyTorch库已被修复。SHA256不是来源认证。

两份示例bundle级别为engineering-demo，未校准、未经过真实科学评价。没有自动resize、统一batch SDK、稳定embedding、微调、分量关联或正式科学权重。

下一编码步骤04：基于bundle提供统一单图/批量SDK与CLI，定义任务输出、样本ID、模型版本及失败清单，验证single/batch一致和仓库外运行。并行数据步骤07—10按 [接入顺序](data-audit-datasets.md#6-后续实施顺序与验收) 推进：版本选择、重复/越界处理、专家叠加复核、源/天空分组、冻结划分、TDD实现PNG/COCO源级适配，之后才训练。当前FITS模型包不能直接接收PNG，不将渲染图伪装成科学FITS；完整源框不能直接用作分量关联监督。
