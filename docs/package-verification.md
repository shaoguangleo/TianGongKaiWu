# 步骤02：安装包验证记录

日期：2026-10-31。结论：**步骤02工程验收通过，最终完整回归125 passed、1 CUDA skipped，351.42s。** 该结论覆盖现有分类/检测软件的打包与安装，不代表科学权重或 CUDA 验收。

## 1. TDD 与回归证据

| 阶段 | 实际行为与结果 | 原始证据 |
|---|---|---|
| 修改前基线 | 123 passed、1 CUDA skipped，35.81s；既有环境，仅作回归起点 | 本轮命令记录；原科学/平台边界不变 |
| 首次 RED | 新测试2 failed：不存在 `tiangong` 包、缺少构建元数据 | [red.txt](evidence/package-2026-10-31/red.txt) |
| 首次 GREEN | 完整回归125 passed、1 CUDA skipped，378.20s；两种产物安装与运行通过 | [pytest.txt](evidence/package-2026-10-31/pytest.txt) |
| 归档缺资料 RED | 审查发现 sdist 的 README 引用未收录；增加实际归档链接检查后1 failed | [archive-red.txt](evidence/package-2026-10-31/archive-red.txt) |
| 修复后最终 GREEN | 补齐引用文档与文本证据后，完整回归125 passed、1 CUDA skipped，351.42s | [pytest-final.txt](evidence/package-2026-10-31/pytest-final.txt) |

测试代码为 `tests/test_package_install.py` 与 `tests/package_smoke.py`。没有 mock 模型、FITS、构建器或安装器。既有测试不改写；11个迁移模块与迁移前代码逐文件比较，除内部 import 改相对导入外一致。

## 2. 安装验收具体做了什么

1. 在源码副本中构建 sdist，再从 sdist 构建 wheel，检查两类产物都生成；源码包中 Markdown 本地引用必须有实际文件。
2. wheel 与 sdist **各使用一个新建 venv**，`include-system-site-packages = false`。后者由 pip 在独立构建环境中重新构建，未借用源码工作区。
3. 从目标产物解析安装声明的依赖，并分别运行 `pip check`，两者均报告 `No broken requirements found`。
4. 将 smoke 脚本复制到仓库外，清除 `PYTHONPATH/PYTHONHOME`，使用目标解释器 `-I` 执行；验证项目和 NumPy/PyTorch/SciPy/Astropy 都来自目标 venv。
5. 写入含负值与 NaN 的真实 FITS 文件，运行读取、mask/asinh、`torch.from_numpy`；验证原文件字节未改、非法科学立方体仍被拒绝。
6. 分类前向输出 `[1,3]`，检测框 `[1,3,4]`、logits `[1,3,2]`，输出有限且框位于归一化范围。
7. 四个安装命令实际生成模拟数据；执行分类训练/预测及检测训练/评价/预测，验证 JSON、类别、样本数与 checkpoint 使用。

此处“真实 FITS”指真实文件格式与读写路径，**像素是工程模拟，不能解释为真实观测训练或性能评价**。模拟训练仅运行最小轮数验证软件链路。

## 3. 环境和可复现材料

实测：macOS 26.6.2、x86_64 Python 3.10.9、CPU；当前主机原生 shell 报 arm64，所以本轮是 x86_64 兼容进程验收，不覆盖原生 arm64。

| 组件 | 实测版本 |
|---|---|
| tiangongkaiwu | 0.1.0.dev0 |
| PyTorch / NumPy | 2.2.2 / 1.26.4 |
| Astropy / SciPy | 6.1.7 / 1.10.0 |
| pytest | 9.2.0 |
| build / setuptools / wheel | 1.6.1 / 84.2.1 / 0.48.0 |

记录包含 [环境与源码摘要](evidence/package-2026-10-31/environment.json)、[核心命令](evidence/package-2026-10-31/commands.txt)、[构建日志](evidence/package-2026-10-31/build.log)、[安装汇总](evidence/package-2026-10-31/installation-report.json)。完整传递依赖分别见 [wheel freeze](evidence/package-2026-10-31/wheel-freeze.txt) 和 [sdist freeze](evidence/package-2026-10-31/sdist-freeze.txt)；它们记录本机安装来源，不是跨平台锁文件。

| 产物 | 安装 | 依赖检查 | 仓库外运行 |
|---|---|---|---|
| wheel | [日志](evidence/package-2026-10-31/wheel-install.log) | [日志](evidence/package-2026-10-31/wheel-check.log) | [日志](evidence/package-2026-10-31/wheel-smoke.log) |
| sdist | [日志](evidence/package-2026-10-31/sdist-install.log) | [日志](evidence/package-2026-10-31/sdist-check.log) | [日志](evidence/package-2026-10-31/sdist-smoke.log) |

使用 [CPU 直接依赖约束](../config/constraints-cpu-tested.txt) 和预下载 wheelhouse 离线执行正式安装测试；下载只获取依赖，未上传或发布任何内容。安装测试没有使用旧的 system-site-packages 环境。

沙箱内新建环境的原生扩展首次加载停滞，沙箱外同环境可以导入；取消了早期沙箱尝试，正式完整验证在沙箱外执行。观察到 NNPACK 硬件不支持警告，CPU 流程仍通过；没有据此给出加速或吞吐保证。

## 4. 实现选择与适用边界

- 依照已批准步骤02在当前工作区实现，保留未跟踪数据与暂存状态，不创建提交。
- 本地发行名 `tiangongkaiwu`、导入名 `tiangong`、版本 `0.1.0.dev0`。没有注册公共索引名称、上传 PyPI 或决定公开分发许可证。
- 根入口使用模块别名，维持私有符号及 patch 语义；旧脚本使用前需安装包，不修改 `sys.path` 掩盖遗漏。
- NumPy 上限为 `<2`，保证本次 PyTorch 2.2.2 的数组互操作；代价是暂不接纳仅支持 NumPy 2 的环境，放宽须另做验证。
- 元数据允许 Python 3.10—3.12；本轮只验证上表组合。其他 Python、原生 arm64、Linux/Windows 和 CUDA 未验证。
- 未新增统一 bundle、embedding、微调或关联；不把本步骤完成等同于阶段 S1 全部完成或科学发布完成。

## 5. 交付与后续

本地 `dist/` 提供 wheel、sdist 与 `SHA256SUMS`。最终文档完成后重建交付产物，核对其 Python 载荷与已验证产物一致，再在上述独立环境中卸载旧包、从最终产物重装、检查并运行 smoke。最终交付摘要及重装结果单独保存在仓库的 `docs/evidence/package-2026-10-31/final-artifacts.json`，不回写已经校验的归档，避免摘要自引用。

[安装说明](installation.md) 提供操作步骤。下一编码任务是统一模型 bundle（步骤03），真实数据确认（步骤07）可并行。
