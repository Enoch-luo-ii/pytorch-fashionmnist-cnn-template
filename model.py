# model.py

"""
模型文件：定义神经网络结构。

这里使用一个简单的 CNN 模型，用于 FashionMNIST / MNIST 图片分类。

输入图片形状：
[batch_size, 1, 28, 28]

输出结果形状：
[batch_size, 10]
"""

import torch
from torch import nn

from config import NUM_CLASSES


class CNN(nn.Module):
    """
    一个简单的卷积神经网络 CNN。

    网络结构：
    Conv2d → MaxPool2d → ReLU
    Conv2d → MaxPool2d → ReLU
    Flatten
    Linear
    """

    def __init__(self):
        super().__init__()

        # 输入通道数是 1，因为 FashionMNIST / MNIST 是灰度图
        # 输出通道数是 10，表示提取 10 个特征图
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=10,
            kernel_size=5
        )

        # 第二个卷积层：输入 10 个通道，输出 20 个通道
        self.conv2 = nn.Conv2d(
            in_channels=10,
            out_channels=20,
            kernel_size=5
        )

        # 最大池化层：把图片尺寸缩小一半
        self.pooling = nn.MaxPool2d(kernel_size=2)

        # 激活函数：增加非线性表达能力
        self.relu = nn.ReLU()

        # 全连接层：把卷积提取到的特征映射到 10 个类别
        # 这里的 320 来自：20 * 4 * 4
        self.fc = nn.Linear(320, NUM_CLASSES)

    def forward(self, x):
        """
        前向传播：定义数据如何通过模型。
        """

        batch_size = x.size(0)

        # 第一组：卷积 → 池化 → 激活
        x = self.conv1(x)
        x = self.pooling(x)
        x = self.relu(x)

        # 第二组：卷积 → 池化 → 激活
        x = self.conv2(x)
        x = self.pooling(x)
        x = self.relu(x)

        # 展平成二维张量：[batch_size, 320]
        x = x.view(batch_size, -1)

        # 输出 10 个类别的 logits
        logits = self.fc(x)

        return logits