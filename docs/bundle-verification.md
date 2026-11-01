# 步骤03验证记录

日期：2026-11-01。最终完整回归通过：**315 passed、2 CUDA skipped（448.96s）**。wheel/sdist各自在新建、不继承系统包的环境中完成安装与真实运行。

## TDD证据

- [bundle初始RED](evidence/bundle-2026-11-01/bundle-red.txt)：先验证接口缺失，再实现双模型往返、坏包拒绝和旧格式迁移。
- [分类RED](evidence/bundle-2026-11-01/classification-red.txt)：版本化checkpoint、冻结清单、原子保存、明确legacy边界。
- [工厂RED](evidence/bundle-2026-11-01/registry-red.txt)：受控结构、参数/资源约束、严格权重检查。
- [计算语义RED](evidence/bundle-2026-11-01/review-red.txt)：norm_first、activation、attention batch_first/num_heads改动必须拒绝。
- [输入边界RED](evidence/bundle-2026-11-01/input-red.txt)：FIFO加载阻塞与非法来源字段；修复后专项结果见 [focused-final.txt](evidence/bundle-2026-11-01/focused-final.txt)。

- [safetensors与信任门槛RED](evidence/bundle-2026-11-01/safetensors-red.txt)：先禁用torch.load证明旧bundle依赖pickle；[最终专项](evidence/bundle-2026-11-01/safetensors-final.txt) **190 passed、1 CUDA skipped**，确认两种bundle的导出/加载均不使用torch.load。

## 完整复测

```bash
TIANGONG_TEST_WHEELHOUSE=/tmp/tiangong-package-wheelhouse \
TIANGONG_TEST_CONSTRAINTS="$PWD/config/constraints-bundle-cpu-tested.txt" \
PIP_CACHE_DIR=/tmp/tiangong-package-pip-cache \
/tmp/tiangong-step02-dev/bin/python -m pytest -q \
  --basetemp=/tmp/tiangong-step03-safe-full-test
```

初版完整回归 **310 passed、2 CUDA skipped（451.54s）**，日志保留为 `pytest-before-safetensors.txt`；安全核对后改用safetensors，最终结果单独记录，不能用初版通过代替最终验证。

完整日志见 [pytest.txt](evidence/bundle-2026-11-01/pytest.txt)。包测试从源码副本构建wheel/sdist，分别安装到不继承系统包的新环境，仓库外以 `python -I` 执行FITS→双模型、五个CLI、训练/评价/预测、两种checkpoint迁移。移走原checkpoint与训练manifest后再次加载bundle，要求预测与原模型逐位一致。

实际环境沿用步骤02验证组合：macOS兼容环境的x86_64 Python3.10.9、PyTorch2.2.2、NumPy1.26.4、SciPy1.10.0、Astropy6.1.7、safetensors0.8.0、CPU。运行测试时原生扩展首次加载在沙箱内停滞，已采用沙箱外执行；依赖来自本地wheelhouse。未验证CUDA、原生arm64、Linux/Windows或其他Python/依赖组合。

## 交付与限制

本步交付工程版本0.1.0.dev1的wheel/sdist和两份engineering-demo `.tgb`，以及各自输入FITS样例。分类包为64×64三类模型，检测包为32×32、1前景类、3queries、hidden_dim=16。源权重来自安装smoke中的一轮模拟训练；导出/恢复逐位一致不代表科学精度。最终归档与模型摘要、安装后复验记录写入 `docs/evidence/bundle-2026-11-01/final-artifacts.json`；该外部记录在归档之后产生，不回写归档造成摘要自引用。

代码审查已复现并修复检测器非权重配置遗漏与FIFO阻塞；安全核对发现旧PyTorch反序列化漏洞，发布bundle已改为safetensors，旧.pt迁移增加显式信任门槛，详见 [边界说明](bundle.md#4-文件契约和完整性)。最终验收使用测试实际输出与产物安装日志，最终分发文件的Python代码摘要必须与全套测试产物一致；最终源码包的文档链接和schema整数上限另做静态复验。没有运行用户约5000张图片的训练，没有认定HeTu示例为真实射电数据，也未发布科学权重。SHA256是完整性检查而非来源认证。
