# PNG/COCO 完整射电源数据接入实施计划

日期：2026-11-01。依据：已确认的 [总体计划](PLAN.md) 07—10及 [数据审计](data-audit-datasets.md)。用户已授权开始，并确认暂缺分组依据时交付未分区候选及模板；开发在macOS，服务器验收目标为Rocky、Python3.12、NVIDIA Tesla V100，驱动/CUDA版本尚未知。

**目标：** 交付可移植的COCO清洗导入、明确分组划分、PNG Dataset、CLI和服务器验收脚本；本轮不训练或声称发布科学模型。

**架构选择：** 保留现有FITS/分量模块，新建源级数据契约与入口。原COCO/PNG只读，派生清单保存摘要、原ID及处理记录。默认严格拒绝异常；显式启用后只合并内容完全一致的重复image记录，其他冲突/异常图片整体隔离待复核，避免漏标目标变成负样本。暂不自动裁边、多边形修复、resize或裁剪。

**技术：** Python标准库、Pillow 12系列（支持项目Python3.10—3.12）、已有NumPy/PyTorch；不增加COCO编译依赖。通过Pillow verify后重新打开并load，完整解码PNG。官方依据：[Pillow图像API](https://pillow.readthedocs.io/en/stable/reference/Image.html)、[Python兼容表](https://pillow.readthedocs.io/en/stable/installation/python-support.html)、[COCO API源代码](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py)。

## 约束与处理决定

- 用户确认标注单位为完整射电源，0:CJ(Core Jet)、1:CS(致密点源)、2:FRI、3:FRII(Fanaroff–Riley)。不改变 `component_detection` 或既有bundle。
- 使用当前未提交的步骤03成果；就地增量实现，先记录文件摘要与索引，不执行git add/reset/commit，不另建缺失这些未提交成果的分支。
- 不复用旧0531划分、不按图片随机划分、不从文件名前缀推断物理源。划分必须由外部表明确给出每个sample的group_id/split和分组依据；程序检查组/字节不跨区，科学天空独立性仍标unverified。
- 默认空标注图隔离或拒绝；只有显式允许才作为空目标样本。存在无效/不支持标注时整图隔离，不静默丢掉某个目标。
- RGBA只允许全不透明alpha；RGB直接缩放到float32 [0,1]，不恢复科学流量。支持8-bit RGB/RGBA PNG，不支持动画、调色板、灰度和非PNG伪装。
- 服务器测试不可在本机冒充通过。macOS的依赖/平台问题与业务逻辑失败分别报告；真实GPU测试以服务器结果为准。

## 文件与接口

| 文件 | 责任 |
|---|---|
| `src/tiangong/png.py` | `read_png(path, *, expected_sha256=None)` 返回 `(CHW float32 ndarray, metadata)`；metadata含width/height/mode/sha256；单文件上限64MiB、像素上限16M |
| `src/tiangong/coco_import.py` | `import_source_coco(coco_path, image_root, *, duplicate_policy="error", invalid_policy="error", allow_empty=False) -> dict` |
| `src/tiangong/source_manifest.py` | `validate_source_manifest(payload) -> dict`、`read_source_manifest(path, *, expected_sha256=None) -> dict`、`write_source_manifest(payload,path)`、`assign_source_splits(payload,assignments,*,grouping_provenance) -> dict` |
| `src/tiangong/source_data.py` | `SourceDataset(manifest_path,image_root,*,split=None,expected_sha256=None)`，返回 `(image, target, sample_id)`；target包含boxes(float32 N×4归一化cxcywh)、labels(int64)、pixel_boxes、polygons、source_ids；`source_collate_fn`保留不同尺寸图片列表 |
| `src/tiangong/source_cli.py` | `prepare`导入、`split`按JSON分组表划分、`inspect`摘要；输出独占写入，错误非零退出；`python -m tiangong.source_cli`及`tiangong-source-data` |
| `scripts/validate_source_data.py` | 服务器验证入口：环境记录、专项/全套pytest、可选CUDA实际前向反向；保留日志与失败，不自动安装GPU依赖 |
| `tests/test_png_source.py`、`test_coco_import.py`、`test_source_manifest.py`、`test_source_data_cli.py`、`test_server_validation.py` | 各模块TDD及CLI/真实数据独立路径测试 |

## 清单v1（字段契约）

顶层字段：`schema_version=1`、`task="source_detection"`、`annotation_unit="complete_radio_source"`、`coordinate_system="pixel_xyxy"`、`preprocessing`、`categories`、`provenance`、`partition`、`samples`、`review`、`changes`、`summary`。

- preprocessing固定 `{version:"png-rgb-unit-v1",channels:3,alpha_policy:"require_opaque",resize:null}`。
- categories按COCO原ID排序，含 `{id,name,label}`；四类名称和ID按上述用户确认定义，label为0—3。
- provenance含 `coco_sha256`、`coco_file_name`、`policy:{duplicate_policy,invalid_policy,allow_empty}`。不写开发者绝对路径。
- partition初始 `{status:"unassigned",grouping_provenance:null,scientific_independence:"unverified"}`；分组后status为`assigned`，必须覆盖train/val/test且各非空。此状态仅指工程划分完成，不代表科学验收。
- 每个sample：`sample_id`、`file_name`（受限相对POSIX路径）、`sha256`、`width/height`、`original_image_ids`、`group_id/split`（初始null）、`sources`。
- 每个source：`source_id`、`category_id`、`label`、`bbox`（pixel xyxy）、`segmentation`（COCO polygon列表，允许缺省为[]的纯bbox标注）、`area`、`original_annotation_ids`。几何有限、面积正、框/多边形在图内；不支持crowd/ignore/RLE，整图报错或隔离。
- `review`记录file_name/image_ids/annotation_ids/reason_codes，`changes`记录合并动作及旧ID。summary记录图片/标注接收数、隔离图片数与逐类计数，并与实际清单重新核对。
- 分组表为JSON列表 `[{sample_id,group_id,split},...]`；拒绝重复/遗漏/多余ID、跨分区group和PNG摘要相同的图片。记录分组表摘要，不声称物理源身份已验证。

实施补充：允许未分区、明确quarantine且review非空的“仅复核”清单，Dataset拒绝空samples。拓扑验证在共享 `source_geometry.py`，每polygon最多1024顶点。review允许安全的非PNG文件名，accepted仍要求PNG。读取时重算分组摘要；清洗policy与changes/review一致；合并annotation ID数量必须匹配image ID数量。

## 分步TDD与交付记录

- [x] A. PNG解码和Dataset：先测RGB/RGBA、透明/损坏/伪装/尺寸与摘要、空框和多目标、源级任务拒绝混用、数据搬迁与batch；看RED，再实现并验证GREEN。
- [x] B. COCO导入：先测类别0、外键/重复ID/非有限数/路径、完全重复可合并、不同标注冲突、整图隔离、越界、空图、polygon与框一致；看RED，再实现GREEN。
- [x] C. 清单与分组：先测版本/任务/标签/几何/摘要、独占原子写入、组/内容泄漏、分组表完整性与原清单不变；看RED，再实现GREEN。
- [x] D. CLI与服务器验收：先测实际子进程prepare→inspect→split→Dataset、失败退出/防覆盖、服务器日志/返回码/缺CUDA；实现后验证。
- [x] E. 用真实result.json生成严格失败证据及显式merge_identical+quarantine的候选清单，完整解码被接收PNG；没有可靠分组表时仅交付未分区清单和可填写模板，不伪造独立test。
- [x] F. 独立审查、必要修复与回归；更新总体计划、handoff、使用说明、schema、服务器命令和验证记录。只将实际完成部分标完成。

专项测试：`python -m pytest -q tests/test_png_source.py tests/test_coco_import.py tests/test_source_manifest.py tests/test_source_geometry.py tests/test_source_data_cli.py tests/test_server_validation.py`。完整回归：`python -m pytest -q`。macOS记录仅为开发验证；Rocky/V100验收命令随服务器脚本交付。

## 重点复核

1. 合并重复时不能丢掉不同标注；隔离图不能以空图进入训练。
2. PNG移动服务器后摘要一致，绝对路径不进入发布清单；链接越界拒绝。
3. RGBA透明数据、损坏IDAT/CRC、动画和超大解码应有明确错误。
4. 人工改manifest导致标签/几何/计数/分区冲突应被检查；已有文件不得被覆盖。
5. 旧模型仍是FITS双通道分量模型；Dataset完成不代表源级模型训练/发布已完成。

最终记录：6文件专项249 passed；完整564 passed、2 CUDA skipped；wheel/sdist干净安装通过。真实候选1999图/2484标注、4图隔离。用户确认暂无分组依据，本轮交付unassigned清单与模板。Rocky/Python3.12/V100验收待服务器执行。
