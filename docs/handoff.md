# TianGongKaiWu 步骤03交接

日期：2026-11-01。状态：步骤03实现完成，最终安装验收中；验收后下一编码步骤为04。

## 本次目标与结论

依据 [统一20步计划](PLAN.md)，在步骤02可安装包基础上完成统一模型bundle，并参考HeTu核对后续真实数据接口。用户报告约有5000张标注图片；目录、格式与类别定义尚未提供，不能据此报告已接入训练集。

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
- 最终完整回归正在执行；日志、实际环境、详细命令见 [bundle-verification.md](bundle-verification.md)。安装测试使用无system-site-packages的新环境，仓库外 `python -I` 运行FITS、双模型、五个CLI；移走checkpoint和训练manifest后恢复bundle，要求预测逐位一致。
- 每次测试样本均为真实FITS格式内的模拟像素，不是用户真实数据或科学验收。未执行git add/reset/commit；既有暂存修改保留。

## 交付路径

最终产物计划位于 `dist/tiangongkaiwu-0.1.0.dev1-py3-none-any.whl`、`dist/tiangongkaiwu-0.1.0.dev1.tar.gz`；两份工程bundle与配套FITS位于 `dist/bundles/`，摘要清单位于 `dist/STEP03-SHA256SUMS`。该目录由Git忽略，不自动提交。最终安装复验和摘要记录另存 `docs/evidence/bundle-2026-11-01/final-artifacts.json`，在归档后形成，不回写已校验归档。

## 适用边界与下一步

本轮实际验证目标是macOS兼容环境的x86_64 Python3.10.9、PyTorch2.2.2、NumPy1.26.4、SciPy1.10.0、Astropy6.1.7、safetensors0.8.0、CPU；不代表原生arm64、CUDA、Linux/Windows或其他版本组合已验收。首次加载原生扩展在沙箱中停滞，测试采用已批准的沙箱外执行，依赖离线安装。

现有分类/检测`.pt`入口及迁移只能处理可信本地实验输出，不能因为设置weights_only而处理来源未知的文件。bundle采用非pickle格式解决发布加载路径问题，不代表旧PyTorch库已被修复。SHA256不是来源认证。

两份示例bundle级别为engineering-demo，未校准、未经过真实科学评价。没有自动resize、统一batch SDK、稳定embedding、微调、分量关联或正式科学权重。

下一编码步骤04：基于bundle提供统一单图/批量SDK与CLI，定义任务输出、样本ID、模型版本及失败清单，验证single/batch一致和仓库外运行。数据步骤07—10可以并行：获得真实目录与格式后，先清点和样例叠加审计，再确定类别映射、源/分量语义、天空分组与冻结划分，最后启动真实训练。
