# 步骤03实施说明

日期：2026-11-01。依据统一计划第12节步骤03；不启动真实数据训练或步骤04批量 SDK。

- 采用单文件 `.tgb` ZIP 模型包，逻辑成员为 config.json、labels.json、preprocessing.json、provenance.json、weights.safetensors、checksums.json。单文件便于完成写入后独占原子发布，拒绝覆盖已有文件；不解压来自用户的路径。
- 仅注册 MorphologyCNN 和 TinySetDetector，结构版本1；明确任务、前景类别顺序、可选原始 category IDs、输入尺寸及双通道 FITS 预处理。权重内绑定元数据摘要，禁止只修改类别/预处理再重算外层摘要后静默加载。
- 当前发布级别只允许 engineering-demo 或 research，不支持把未科学验收的包标为正式科学 release。摘要用于完整性，不是来源签名。
- 导出、加载、显式旧 checkpoint 迁移和 inspect CLI 为本步能力。旧分类文件缺预处理版本时必须由调用者显式确认，未知来源项保留缺失记录；HeTu TensorFlow/Detectron2 权重不自动转换。
- 安全核对后将发布权重改为safetensors，不调用torch.load。旧.pt迁移需显式trusted_checkpoint=True / --trust-checkpoint；现有训练checkpoint入口仍只限可信本地实验文件，不能当作处理未知pickle的安全边界。相关PyTorch漏洞和格式选择见bundle.md。
- 分类链补齐版本化 checkpoint、严格恢复、原子保存、输出防覆盖和冻结 manifest 字节检查；保持检测既有保护。
- TDD：先写双模型 round-trip、坏包/错契约/缺资产、unsafe pickle、原子失败和迁移测试，观察 RED，再实现与完整回归；从实际安装包验证新接口。
- 约5000张标注图片为用户报告的数据规模，用户已确认PNG + Label Studio导出COCO JSON；实际目录、类别语义、是否有配对FITS与空间划分仍待核实。不能将 HeTu 示例目录自动当作这批数据。
