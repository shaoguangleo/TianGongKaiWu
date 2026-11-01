# TianGongKaiWu 步骤02交接

日期：2026-10-31。状态：步骤02已完成本步工程验收；下一编码步骤为03。

## 本次目标

按 [统一计划](PLAN.md) 步骤02，将现有分类与检测项目整理为可安装的 Python 包，遵循 TDD，验证实际分发产物并保留旧入口行为。

## 已完成的实现

- 本地发行名 `tiangongkaiwu`，导入名 `tiangong`，工程版本 `0.1.0.dev0`；新增 `pyproject.toml` 与 `MANIFEST.in`。
- 11个实现模块迁入 `src/tiangong/`，除项目内相对导入外保持原实现。根文件用模块别名保留旧导入、私有函数及 monkeypatch 行为。
- 安装命令：`tiangong-classify`、`tiangong-detect`、`tiangong-demo`、`tiangong-demo-detection`；没有扩张到步骤04的统一 SDK。
- 运行依赖与 `dev` extra 分开；旧 `requirements.txt` 转发到开发安装。NumPy 上限收紧为 `<2`，保留已验证 PyTorch 2.2.2 的数组互操作。
- 增加产物安装测试与独立 smoke 脚本、安装说明和验证记录；源码包补齐 README 引用资料，wheel 只安装 `tiangong` 命名空间。

## TDD 与验证

- 先运行新增测试，观察2 failed：无 `tiangong` 包、无可构建项目元数据。
- 首次完整回归：125 passed、1 CUDA skipped，378.20s；wheel/sdist 各在新建且不继承系统包的环境安装。
- 仓库外 `python -I` 验证 FITS→Tensor、分类/检测前向，以及已安装 CLI 的分类训练/预测、检测训练/评价/预测。样本为真实 FITS 格式中的模拟像素，不是真实巡天科学验收。
- 审查发现 sdist 缺失部分文档引用；增加归档链接检查并观察 RED，再补齐 MANIFEST。修复后完整复测 **125 passed、1 CUDA skipped，351.42s**；新归档检查及 wheel/sdist 的安装与运行均通过。
- 新环境原生扩展在沙箱中首次加载停滞，沙箱外正常；依赖预先下载到临时 wheelhouse，正式验证仅使用本地依赖。
- 完整命令、环境、日志和最终状态见 [打包验证](package-verification.md)。本轮未修改既有测试、约束、模拟数据或历史检测证据；未执行 git add/reset/commit。

## 支持范围与限制

实测组合为 macOS、x86_64 Python 3.10.9、PyTorch 2.2.2、NumPy 1.26.4、SciPy 1.10.0、Astropy 6.1.7、CPU。解释器运行于当前 arm64 主机的兼容环境，不能据此宣称原生 arm64 已验收。观察到 NNPACK 硬件不支持警告，CPU 流程仍通过；未作性能保证。

元数据允许 Python 3.10—3.12；其他 Python、Linux/Windows、原生 arm64 和 CUDA 尚未实际验证。包名未在公共索引注册，未上传 PyPI，许可证与正式托管方式待后续锁定。

旧脚本需先安装项目。步骤02不新增统一 bundle、embedding、微调、分量关联或科学权重；现有 checkpoint 仍是任务专属格式。

## 下一步

进入步骤03：冻结统一 bundle schema，绑定架构、类别顺序、预处理、权重与来源；先测试导出/恢复一致性、损坏/不兼容拒绝和旧 checkpoint 显式迁移，再实现受控工厂与加载器。步骤07的真实数据确认可并行推进。

[安装与运行说明](installation.md) 提供开发安装、wheel/sdist 安装、四个入口、Python 调用和离线复测方法。根目录 `handoff.md` 继续作为本文导航。

## 本地交付物

`dist/` 中提供 `tiangongkaiwu-0.1.0.dev0-py3-none-any.whl`、`tiangongkaiwu-0.1.0.dev0.tar.gz` 和 `SHA256SUMS`。生成物目录被 Git 忽略，源码与文档由用户手工提交。最终产物重装与运行的记录单独保存在 `docs/evidence/package-2026-10-31/final-artifacts.json`，不回写已经校验的归档。
