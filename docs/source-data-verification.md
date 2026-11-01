# PNG/COCO源级接入验证记录

日期：2026-11-01；开发版 `0.1.0.dev2`。源码、测试、CLI、JSON schema及服务器脚本均已落盘，实际结果持续记录于 `docs/evidence/source-data-2026-11-01/`。

## 1. TDD与独立复核

PNG/Dataset、COCO、清单/分组、CLI/服务器脚本分别先记录缺失行为的RED，再实现GREEN。新增几何验证器及复核问题也有各自RED/GREEN；包括截断zlib、面积非零的自相交轮廓、特殊文件阻塞、重复JSON键、浮点指数溢出、清洗策略来源不一致、划分摘要漂移、隔离ID重叠。中间的RED、环境阻塞与主动中止日志保留，不算最终通过。

第一次集成专项（6文件）为241 passed；独立复核增加测试与修复后，经真实服务器脚本入口执行的最终专项为 **249 passed（8.42s）**，见 [专项日志](evidence/source-data-2026-11-01/macos-server-entry-final/pytest.log)。独立审查最后12项复核通过，没有剩余已确认P2问题。测试覆盖安装之外的真实子进程CLI和可迁移图片根目录，不要求测试机器具备原始HeTu数据。

## 2. 本机完整回归

命令（项目根目录）：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
TIANGONG_TEST_WHEELHOUSE=/tmp/tiangong-package-wheelhouse \
TIANGONG_TEST_CONSTRAINTS="$PWD/config/constraints-source-cpu-tested.txt" \
/tmp/tiangong-step02-dev/bin/python -m pytest -q
```

首次full因本机磁盘空间不足：**1 failed、516 passed、2 skipped、47 errors**，详见 [失败日志](evidence/source-data-2026-11-01/pytest-full-disk-error.txt)。失败/错误涉及 `test_wheel_and_sdist_run_in_clean_environments` 与 `test_png_source.py` 的临时文件创建，底层为 `OSError: [Errno 28] No space left on device`。不将该结果隐藏或作为通过。

仅清理前序本任务安装测试生成的9个wheel/sdist临时虚拟环境，保留开发环境、wheelhouse、输入、源码、dist、日志和源副本。[清理记录](evidence/source-data-2026-11-01/temp-cleanup.json)。最终完整回归 **564 passed、2 CUDA skipped（435.15s）**，见 [最终full日志](evidence/source-data-2026-11-01/pytest-full.txt)。新增源码、测试及配置与受测sdist中的45个文件字节一致，见 [归档源码校验](evidence/source-data-2026-11-01/archive-source-check.json)。

## 3. 真实数据结果

原始JSON SHA256：`bf7209e1fa254b8fcb023cec21182334fb22c82ce0b4862c651623077ab322cd`。严格导入非零退出且未写输出；显式 `merge_identical`＋`quarantine` 导入成功，完整解码接收图片，未改动原件。

- 候选：1999张PNG、2484条标注，CJ376 / CS1653 / FRI72 / FRII383。
- 合并：2组完全一致重复身份（image639/640/641及1582/1583）。
- 隔离：4张不同图片，共5条原image记录、10条annotation记录。包括image38的自相交/相触轮廓（ann83）；image1844/1845标注差异；image1879、1889含越界。整图所有标注保留待复核。
- 原2499条annotation减去5条完全重复记录及10条隔离记录，得到2484条接收标注；不把记录数等同独立物理源数。
- 清单unassigned，附null分组模板；缺少源/天空对应表，不生成虚假的独立test。

当前代码重新生成的候选与落盘内容完全一致；实际Dataset读取1999样本，首批16张均为3×150×150，见 [真实Dataset复验](evidence/source-data-2026-11-01/real-dataset-smoke.json)。

原始日志与统计见 [real-strict.txt](evidence/source-data-2026-11-01/real-strict.txt)、[real-prepare.txt](evidence/source-data-2026-11-01/real-prepare.txt)、[real-import.json](evidence/source-data-2026-11-01/real-import.json)。制品位于 `dist/source-data/`，由Git忽略，源数据只按相对路径引用，没有复制到分发包。

## 4. 服务器验收边界

服务器目标是Rocky、Python3.12、NVIDIA Tesla V100；没有实际服务器执行证据。脚本保存环境、驱动查询、pytest日志/退出码和真实CUDA矩阵乘法反向传播结果。macOS不能通过 `--require-cuda`，这一行为已用真实调用与测试确认；不将其解释为V100存在故障。

最终macOS脚本复验：pytest退出0、249项通过；CUDA检查因不可用失败，脚本整体非零，见 [脚本结果](evidence/source-data-2026-11-01/macos-server-entry-final/result.json) 和 [环境记录](evidence/source-data-2026-11-01/macos-server-entry-final/environment.json)。这验证了失败不被伪装成功，没有验证V100。

本机Python3.10.9 / x86_64兼容环境 / PyTorch2.2.2 / Pillow12.3.0 / CPU仅用于开发验证；服务器Python3.12、驱动/CUDA/架构兼容及训练性能必须实测。服务器具体命令见 [使用说明](source-data.md#6-tdd及rockyv100验收)。尚未实现源级训练入口、源级bundle、科学归一化、天空隔离或科学模型发布。

## 5. 分发制品与工作区

`dist/tiangongkaiwu-0.1.0.dev2-py3-none-any.whl` 与 `dist/tiangongkaiwu-0.1.0.dev2.tar.gz` 来自上述实际通过安装测试的产物；候选与模板位于 `dist/source-data/`。四份制品摘要在 `dist/SOURCE-DATA-SHA256SUMS`，另见 [制品记录](evidence/source-data-2026-11-01/final-artifacts.json)。wheel/sdist均在不继承系统包的新环境中安装，仓库外 `python -I` 验证既有双模型流程、新CLI prepare/inspect/split及搬迁后的Dataset，原始类别0保留。详见 [安装报告](evidence/source-data-2026-11-01/package-installation-report.json)；NNPACK在此macOS兼容硬件上报告不支持，实际CPU流程仍通过。

原始3份COCO、4014条PNG引用与6份步骤03产物摘要不变，见 [输入保全](evidence/source-data-2026-11-01/input-preservation.json)。用户确认缺少可靠源/天空对应表，本轮按授权交付unassigned候选和模板；没有生成真实科学划分、运行服务器测试或训练新权重。工作区最终检查见 [final-workspace-check.json](evidence/source-data-2026-11-01/final-workspace-check.json)。

收尾发现暂存区与本轮起始快照不同（期间有外部暂存/版本操作）；本助手未执行git add/reset/commit，保留当前索引，不回退。运行源码仍以受测归档逐文件核对，文档最终补录的工作区差异保留供用户提交。
