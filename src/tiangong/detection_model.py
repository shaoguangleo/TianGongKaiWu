"""从随机权重训练的单尺度集合检测器，不下载任何外部权重。"""

import math
from typing import Dict

import torch
from torch import Tensor, nn


class TinySetDetector(nn.Module):
    """CNN、二维位置编码、learned queries 与 Transformer decoder 基线。

    最后一类为背景；框输出为归一化 ``cx, cy, width, height``。
    这是工程检测基线，输出框不代表物理源边界或科学测量。
    """

    def __init__(self, num_classes: int, num_queries: int = 10, hidden_dim: int = 64) -> None:
        super().__init__()
        for name, value in (("num_classes", num_classes), ("num_queries", num_queries),
                            ("hidden_dim", hidden_dim)):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} 必须是整数")
            if value <= 0:
                raise ValueError(f"{name} 必须大于 0")
        if hidden_dim % 4 != 0:
            raise ValueError("hidden_dim 必须能被 4 整除，以支持四头注意力和二维位置编码")
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        # 保留空间布局；GroupNorm 不依赖 batch 大小或图像空间像素数。
        layers = []
        channels = (2, 16, 32, hidden_dim)
        for input_channels, output_channels in zip(channels[:-1], channels[1:]):
            layers.extend([
                nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(1, output_channels),
                nn.ReLU(),
            ])
        self.backbone = nn.Sequential(*layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.queries = nn.Embedding(num_queries, hidden_dim)
        self.class_head = nn.Linear(hidden_dim, num_classes + 1)
        self.box_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

    def _position_encoding(self, height: int, width: int, reference: Tensor) -> Tensor:
        """每个坐标轴分配一半维数，支持非正方形与单像素特征图。"""
        quarter_dim = self.hidden_dim // 4
        frequencies = torch.pow(
            reference.new_tensor(10000.0),
            torch.arange(quarter_dim, device=reference.device, dtype=reference.dtype) / quarter_dim,
        )
        x = torch.arange(width, device=reference.device, dtype=reference.dtype)
        y = torch.arange(height, device=reference.device, dtype=reference.dtype)
        x = x[:, None] * (2.0 * math.pi / max(width - 1, 1)) / frequencies
        y = y[:, None] * (2.0 * math.pi / max(height - 1, 1)) / frequencies
        x_encoding = torch.cat((x.sin(), x.cos()), dim=-1)
        y_encoding = torch.cat((y.sin(), y.cos()), dim=-1)
        grid = torch.cat((
            x_encoding[None, :, :].expand(height, -1, -1),
            y_encoding[:, None, :].expand(-1, width, -1),
        ), dim=-1)
        return grid.reshape(1, height * width, self.hidden_dim)

    def forward(self, images: Tensor) -> Dict[str, Tensor]:
        """接受有限浮点 ``[B, 2, H, W]``，返回分类 logits 和归一化框。"""
        if not isinstance(images, Tensor) or not images.is_floating_point():
            raise TypeError("images 必须是浮点 Tensor")
        if images.ndim != 4 or images.shape[1] != 2 or any(size <= 0 for size in images.shape):
            raise ValueError("images 必须为非空 [B, 2, H, W]")
        if not torch.isfinite(images).all():
            raise ValueError("images 包含非有限值")
        features = self.backbone(images)
        batch_size, _, height, width = features.shape
        memory = features.flatten(2).transpose(1, 2)
        memory = memory + self._position_encoding(height, width, memory)
        queries = self.queries.weight.unsqueeze(0).expand(batch_size, -1, -1)
        decoded = self.decoder(queries, memory)
        return {"logits": self.class_head(decoded), "boxes": self.box_head(decoded).sigmoid()}
