

# dataset.py

"""
数据加载文件：负责下载数据集，并创建 DataLoader。

Dataset 负责存储数据样本和标签。
DataLoader 负责按 batch 批量读取数据。
"""

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from config import DATA_DIR, BATCH_SIZE


def get_dataloaders(batch_size=BATCH_SIZE):
    """
    下载并加载 FashionMNIST 数据集。

    返回：
    - train_dataloader: 训练集 DataLoader
    - test_dataloader: 测试集 DataLoader
    """

    training_data = datasets.FashionMNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=ToTensor()
    )

    train_dataloader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False
    )

    return train_dataloader, test_dataloader


def check_dataloader_shape(dataloader):
    """
    查看一个 batch 的图片和标签形状。
    """

    for X, y in dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break