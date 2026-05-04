

# utils.py

"""
工具函数文件：存放项目中可以复用的小函数。

这里主要包含：
- 自动选择训练设备
- 保存模型参数
- 加载模型参数
"""

import torch


def get_device():
    """
    自动选择训练设备。

    优先级：
    1. cuda：NVIDIA GPU
    2. mps：Mac Apple Silicon 加速
    3. cpu：普通 CPU
    """

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Using {device} device")
    return device


def save_model(model, path):
    """
    保存模型参数。

    注意：这里保存的是 state_dict，也就是模型学到的参数，
    不是整个模型结构。
    """

    torch.save(model.state_dict(), path)
    print(f"Saved PyTorch Model State to {path}")


def load_model(model, path, device):
    """
    加载模型参数。

    使用这个函数时，需要先创建和保存时相同结构的模型，
    再把参数加载进去。
    """

    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Loaded PyTorch Model State from {path}")
    return model