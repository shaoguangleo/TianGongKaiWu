"""模型层：自定义小CNN，无预训练权重、不依赖DINO/AION。"""
import torch.nn as nn


class MorphologyCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 第1段：卷积提取局部形态，池化逐步扩大感受野。
        blocks = []
        channels = [2, 16, 32, 64]
        for a,b in zip(channels[:-1],channels[1:]):
            blocks += [nn.Conv2d(a,b,3,padding=1), nn.ReLU(), nn.MaxPool2d(2)]
        self.encoder = nn.Sequential(*blocks)
        # 第2段：保留4×4粗空间布局，有助于区分紧致与双叶；这不是旋转不变网络。
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d((4,4)), nn.Flatten(),
                                  nn.Linear(64*4*4,64), nn.ReLU(),
                                  nn.Linear(64,num_classes))

    def forward(self, x):
        # 第3段：训练时返回logits；交叉熵内部处理归一化，不能重复softmax。
        return self.head(self.encoder(x))
