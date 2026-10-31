"""生成工程检测模拟数据；高斯分量及其框不是科学真值或真实 RACS 数据。"""
import argparse
import json
from numbers import Integral
from pathlib import Path

import numpy as np
from astropy.io import fits


def _integer(name, value, minimum, maximum=None):
    """布尔值不是有效计数；先校验参数，避免坏参数留下输出文件。"""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} 必须为整数")
    if value < minimum or (maximum is not None and value > maximum):
        limit = f"[{minimum}, {maximum}]" if maximum is not None else f">={minimum}"
        raise ValueError(f"{name} 必须满足 {limit}")
    return int(value)


def generate_demo(output, per_split=12, size=64, seed=7) -> Path:
    """写出三个分区的 FITS 与 manifest.json，返回清单路径。

    每个分区按 i % 4 包含 0–3 个分量；seed 为 [0, 2**32-1] 的整数。
    同一 NumPy/Astropy 环境中，相同参数生成相同字节。输出必须不存在或
    是空目录；已有文件、非空目录及输出路径本身的符号链接均拒绝使用。
    框使用连续像素边界 xyxy，取生成高斯中心的正负 3 sigma。
    """
    per_split = _integer("per_split", per_split, 4)
    size = _integer("size", size, 32)
    seed = _integer("seed", seed, 0, 2**32 - 1)
    out = Path(output).expanduser()
    if out.is_symlink() or (
        out.exists() and (not out.is_dir() or any(out.iterdir()))
    ):
        raise FileExistsError(f"输出路径已有内容或为符号链接，请使用新目录：{out}")
    out.mkdir(parents=True, exist_ok=True)

    # 像素中心是边界坐标中的 (x+0.5, y+0.5)，不伪造 WCS/波束信息。
    yy, xx = np.mgrid[:size, :size]
    xx, yy = xx + 0.5, yy + 0.5
    rng = np.random.default_rng(seed)
    samples = []
    for split in ("train", "val", "test"):
        for index in range(per_split):
            sample_id = f"{split}_{index:04d}"
            image = rng.normal(0.0, 1.0, (size, size))
            components = []
            # 随机选不同网格；有限抖动与宽度保证 3 sigma 框分离且不出界。
            cells = rng.choice(4, size=index % 4, replace=False)
            for component_index, cell in enumerate(cells):
                cx = (0.25 + 0.5 * (int(cell) % 2)) * size
                cy = (0.25 + 0.5 * (int(cell) // 2)) * size
                cx, cy = np.array([cx, cy]) + rng.uniform(-0.035, 0.035, 2) * size
                sigma_x, sigma_y = rng.uniform(0.025, 0.045, 2) * size
                amplitude = rng.uniform(9.0, 15.0)
                image += amplitude * np.exp(
                    -0.5 * (((xx - cx) / sigma_x) ** 2 + ((yy - cy) / sigma_y) ** 2)
                )
                box = [
                    float(max(0.0, cx - 3.0 * sigma_x)),
                    float(max(0.0, cy - 3.0 * sigma_y)),
                    float(min(size, cx + 3.0 * sigma_x)),
                    float(min(size, cy + 3.0 * sigma_y)),
                ]
                components.append({
                    "component_id": f"{sample_id}_component_{component_index}",
                    "label": 0,
                    "box": box,
                })
            relative_path = f"{sample_id}.fits"
            header = fits.Header()
            header["SYNTHET"] = (True, "Engineering simulation; not scientific truth")
            # Astropy 默认不覆盖；即使生成时发生文件名冲突也不能覆盖原文件。
            fits.writeto(out / relative_path, image.astype(np.float32), header, overwrite=False)
            samples.append({
                "sample_id": sample_id,
                "path": relative_path,
                "split": split,
                "group_id": f"synthetic_{sample_id}",
                "components": components,
            })

    manifest = out / "manifest.json"
    document = {
        "schema_version": 1,
        "coordinate_system": "pixel_xyxy",
        "classes": ["radio_component"],
        "synthetic": True,
        "note": "仅用于工程验证；高斯分量框不是科学真值，不模拟真实观测响应。",
        "generation": {"per_split": per_split, "size": size, "seed": seed},
        "samples": samples,
    }
    # 清单最后发布；独占创建防止覆盖其他进程刚写出的同名文件。
    with manifest.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="demo_detection_data")
    parser.add_argument("--per-split", type=int, default=12)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7, help="整数，范围 [0, 2**32-1]")
    args = parser.parse_args()
    try:
        manifest = generate_demo(args.output, args.per_split, args.size, args.seed)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"写出 {3 * args.per_split} 个工程模拟样本：{manifest}")


if __name__ == "__main__":
    main()
