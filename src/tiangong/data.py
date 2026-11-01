"""数据层：保持原FITS不变，将科学数据转换为训练张量。"""
import csv
import hashlib
import io
from pathlib import Path
import numpy as np
from astropy.io import fits
import torch
from torch.utils.data import Dataset


def read_image(path):
    # 第1段：只读取主HDU；只压缩长度为1的额外维度，绝不静默选取频率平面。
    with fits.open(path, memmap=False) as hdus:
        if hdus[0].data is None:
            raise ValueError(f'{path}: 主HDU没有图像，请先明确选择扩展HDU')
        image = np.asarray(hdus[0].data, dtype=np.float32).squeeze()
    if image.ndim != 2:
        raise ValueError(f'{path}: 需要二维切片，实际为 {image.shape}')
    return image


def preprocess(image):
    # 第2段：有效像素掩码独立作为通道；NaN既不是背景，也不是负流量。
    valid = np.isfinite(image)
    if not valid.any():
        raise ValueError('图像没有有效像素')
    # float32 极值的中位数/方差归约可能溢出；用 float64 统计，再输出 float32。
    values = image[valid].astype(np.float64)
    background = np.median(values)
    # 第3段：MAD估计稳健尺度，退化时用标准差；常数图用1作为安全尺度。
    # 这是切片级工程归一化，不是专业局部RMS测量，也不保留绝对流量。
    scale = float(1.4826 * np.median(np.abs(values-background)))
    if scale < 1e-12:
        scale = float(np.std(values))
    if not np.isfinite(scale) or scale < 1e-12:
        scale = 1.0
    signal = np.zeros_like(image, dtype=np.float32)
    signal[valid] = np.arcsinh(np.clip((values-background)/scale, -1e4, 1e4))
    return np.stack([signal, valid.astype(np.float32)])


def read_manifest(path, num_classes, *, expected_sha256=None):
    # 第4段：CSV是样本的唯一入口；路径相对CSV，而不是当前工作目录。
    path = Path(path).resolve()
    # 从同一批bytes核对冻结摘要并解析，避免检查后再次读盘引入不同清单。
    content = path.read_bytes()
    if expected_sha256 is not None and hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError('manifest SHA256与冻结benchmark不一致，禁止更改样本或split')
    with io.StringIO(content.decode('utf-8-sig'), newline='') as f:
        reader = csv.DictReader(f)
        required = {'sample_id','path','label','split','group_id'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'清单需要字段 {sorted(required)}')
        rows = list(reader)
    if not rows:
        raise ValueError('空清单')
    ids, groups, paths = set(), {}, {}
    for row in rows:
        row['label'] = int(row['label'])
        if not 0 <= row['label'] < num_classes:
            raise ValueError('label超出类别范围')
        if row['split'] not in {'train','val','test'}:
            raise ValueError('split必须为train/val/test')
        if not row['sample_id'].strip() or row['sample_id'] in ids:
            raise ValueError('sample_id为空或重复')
        ids.add(row['sample_id'])
        if not row['group_id'].strip():
            raise ValueError('group_id不能为空')
        if groups.setdefault(row['group_id'],row['split']) != row['split']:
            raise ValueError('group_id跨分区，存在泄漏')
        row['path'] = str((path.parent / row['path']).resolve())
        if paths.setdefault(row['path'], row['split']) != row['split']:
            raise ValueError('同一路径跨分区')
    return rows


class RadioDataset(Dataset):
    """第5段：按需读盘，返回[2,H,W]张量、整数标签、样本编号。"""
    def __init__(self, rows, size):
        self.rows, self.size = rows, size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = read_image(row['path'])
        # 不隐式缩放/裁剪，避免截掉射电叶或改变角尺度却不记录。
        if image.shape != (self.size,self.size):
            raise ValueError(f"{row['sample_id']}: 需要{self.size}方形切片，实际{image.shape}")
        return torch.from_numpy(preprocess(image)), row['label'], row['sample_id']
