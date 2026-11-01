# TianGongKaiWu：射电分类与分量检测工程包

这是自定义 CNN 分类与最小集合检测的工程基线，本地发行名 `tiangongkaiwu`，Python 导入名 `tiangong`。实现统一位于 `src/tiangong/`；根目录同名脚本保留为安装后的兼容入口。尚未发布到 PyPI，也不包含经过真实科学验收的权重。
范围：已裁剪、完整源、固定尺寸的FITS分类。没有实现DINO、AION、大图检测、物理分量关联、FR分类或自动空间划分。

下一阶段已新增独立的最小分量检测器入口 `detect.py`，支持可变数量标注框、Hungarian 匹配、框回归及训练/评估/预测；分类入口保持独立。安装、标注协议、运行命令和边界见 [分量检测器说明](docs/detection.md)，实施与验证见 [检测方案](docs/detection-plan.md) 和 [本次检测验证](docs/detection-verification.md)。

## 1. 安装与快速运行

先使用 Python 3.10—3.12 新建虚拟环境，在仓库根目录安装；实际验证的平台和版本以 [安装说明](docs/installation.md) 与 [打包验证](docs/package-verification.md) 为准。模型默认使用 CPU。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python demo.py --output demo_data --per-class 40
python main.py train --manifest demo_data/manifest.csv --epochs 12 --output runs
python main.py evaluate --manifest demo_data/manifest.csv --checkpoint runs/best.pt --output runs
python main.py predict --image demo_data/test_0_0.fits --checkpoint runs/best.pt --output runs
python -m pytest -q
```

安装后可在仓库外使用 `tiangong-classify`、`tiangong-detect`、`tiangong-demo` 和 `tiangong-demo-detection`，其参数与对应旧脚本一致。构建和从 wheel/sdist 安装的命令见 [安装说明](docs/installation.md)。

代码接受 `--device cuda`，但 CUDA 尚未完成实际平台验证。默认 CPU，训练不下载权重。依赖范围不是跨平台锁文件；早期分类记录保留在 `VERIFICATION.md`，本轮包验证单独记录。
生成器对已有manifest拒绝覆盖。不要使用相同output目录混合不同真实实验；每次运行用新目录。

## 2. 按数据流理解每一段代码

### data.py：从FITS到张量

- `read_image`：读取主HDU，转float32；只移除长度为1的维度。真正的多频率/多偏振立方体报错，需要用户先明确选面。
- `preprocess`第一段：识别有限像素，为NaN/Inf生成掩码，拒绝全无效图。
- `preprocess`第二段：用中位数估计背景，用MAD估计尺度，退化时用标准差或1。asinh压缩动态范围、保留背景扣除后的正负值；极端值裁剪至正负1e4倍尺度。
- `read_manifest`：读CSV，检查类别、唯一ID、group_id跨分区和相同路径跨分区。相对路径以CSV目录为根解析。
- `RadioDataset`：按需读盘、检查尺寸，输出 `(image, label, sample_id)`。`image`形状是 `[2,H,W]`，通道0为归一化信号、通道1为有效掩码。

逐切片归一化会移除绝对流量信息，MAD也可能受大面积弥散源影响。本版仅用于形态基线；下一版应引入独立局部噪声图、波束与流量元数据。原始FITS不被修改。

### model.py：图像到类别分数

- 三组 `Conv2d → ReLU → MaxPool`：学习局部结构并扩大感受野，通道数2→16→32→64。
- `AdaptiveAvgPool2d(4,4)`：压缩成固定空间网格，保留粗布局。
- `Linear → ReLU → Linear`：从空间特征预测类别分数。
- `forward`：返回logits，不在训练前softmax。

默认三类标签：0 compact（紧致/未分辨）、1 extended（单体延展）、2 multi_component（多分量）。这是入门标签；真实数据需专家制定一致规则，尤其核/叶完整性。该CNN未实现旋转不变性、增强或波束条件化，不宣称架构创新。

### engine.py：训练和评价

- `classification_metrics`：构建混淆矩阵并计算accuracy、逐类precision/recall/F1、macro-F1和support。缺失类F1记0，因此必须结合support解读。
- `run_epoch`第一段：有optimizer时切到训练模式，否则切到评估模式并关闭梯度。
- 第二段：前向计算与交叉熵；训练时清梯度、反传、更新参数。
- 第三段：输出softmax分数和逐样本预测。分数未经概率校准，不是科学置信度。

### main.py：组织实验

1. 解析命令与参数、固定随机种子；跨硬件并不保证逐位相同。
2. 评估和预测从checkpoint读取类别顺序及尺寸，防止输出错位。
3. `predict`复用同一预处理，将单图概率写入JSON。
4. 检查全清单后选择split，保证不会只检查局部而漏掉组跨分区。
5. `train`运行train/val，按验证集macro-F1保存最佳权重。若并列保留较早模型；test不参与选优。
6. checkpoint记录权重、类别、尺寸、随机种子、epoch、CSV的SHA256；另存config和history。
7. `evaluate`只读取test并保存指标、混淆矩阵和逐样本结果。

只加载可信checkpoint。CSV摘要不等于图像字节版本管理；正式研发需额外保存图像校验和、代码提交与环境版本。

### demo.py：没有真实数据时验证工程链路

随机生成紧致高斯、延展高斯和双高斯，加入独立噪声与随机旋转，分别写出train/val/test。它们不模拟真实ASKAP波束、uv覆盖、噪声相关性或旁瓣，不能作为科学性能证据。

## 3. 接入真实RACS-mid

本程序不会自动下载或裁剪RACS-mid。准备包含完整物理源的同尺寸切片，保留原始文件及WCS。根据源角尺度选择视野，不要为符合默认64像素截掉外叶；可通过 `--size 256` 等改变输入尺寸。

CSV示例（路径相对CSV）：

```csv
sample_id,path,label,split,group_id
source_0001,cutouts/source_0001.fits,0,train,sky_block_A
source_0002,cutouts/source_0002.fits,1,val,sky_block_B
source_0003,cutouts/source_0003.fits,2,test,sky_block_C
```

真实实验需要每类充足的训练、验证和测试样本，上面三行仅演示格式。自定义类别时使用 `--classes compact extended multi_component`，标签整数按此顺序从0开始。

**group_id必须由你们预先定义**：同一物理源、重叠视野、重复观测、关联天空区域及不同分辨率版本归同组。该程序不能从组名验证真实空间隔离，也无法发现路径不同但内容相同的图像。必须另做坐标、重叠和缓冲区审计。

本版严格拒绝自动缩放，避免无记录地改变角尺度。训练集和推理集应保持一致的数据制作规范。beam/WCS未输入网络；没有提供流量或角尺度测量功能。

## 4. 输出文件

| 文件 | 内容 |
|---|---|
| best.pt | 按验证集选优的模型及基本元数据 |
| config.json | 本次训练参数 |
| history.json | 每轮训练/验证损失与分类指标 |
| test_metrics.json | 独立测试指标与混淆矩阵 |
| test_predictions.json | 独立测试逐样本结果 |
| prediction.json | 单FITS预测分数 |

## 5. 下一阶段

先审查真实错误样本、确定标签和视野，再增加数据增强、观测元数据与噪声图。分类链路稳定后新增检测头和关联任务；最后比较AION多模态分支。此版本不应直接部署为科学目录生产服务。
