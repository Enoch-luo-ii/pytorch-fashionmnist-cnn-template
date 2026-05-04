

# train.py

"""
主训练文件：负责把整个 PyTorch 训练流程串起来。

完整流程：
1. 选择 device
2. 加载数据
3. 创建模型
4. 定义损失函数
5. 定义优化器
6. 训练模型
7. 测试模型
8. 保存模型
"""

import torch
from torch import nn

from config import EPOCHS, LEARNING_RATE, MODEL_SAVE_PATH
from dataset import get_dataloaders, check_dataloader_shape
from model import CNN
from utils import get_device, save_model


def train_one_epoch(dataloader, model, loss_fn, optimizer, device):
    """
    训练一个 epoch。
    """

    size = len(dataloader.dataset)

    # 切换到训练模式
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        # 把数据移动到 device 上
        X, y = X.to(device), y.to(device)

        # 1. 前向传播：模型进行预测
        pred = model(X)

        # 2. 计算 loss：预测结果和真实标签之间的差距
        loss = loss_fn(pred, y)

        # 3. 清空上一轮残留的梯度
        optimizer.zero_grad()

        # 4. 反向传播：根据 loss 计算梯度
        loss.backward()

        # 5. 更新模型参数
        optimizer.step()

        # 每 100 个 batch 打印一次 loss
        if batch % 100 == 0:
            loss_value = loss.item()
            current = batch * len(X)
            print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, device):
    """
    在测试集上评估模型。
    """

    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    # 切换到评估模式
    model.eval()

    test_loss = 0
    correct = 0

    # 测试阶段不需要计算梯度
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)

            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    accuracy = 100 * correct

    print(
        f"Test Error:\n"
        f"Accuracy: {accuracy:>0.1f}%, "
        f"Avg loss: {test_loss:>8f}\n"
    )

    return accuracy, test_loss


def main():
    """
    主函数：运行完整训练流程。
    """

    # 1. 获取训练设备
    device = get_device()

    # 2. 加载训练集和测试集
    train_dataloader, test_dataloader = get_dataloaders()

    # 3. 打印一个 batch 的形状，检查数据是否正确
    check_dataloader_shape(test_dataloader)

    # 4. 创建模型，并移动到 device 上
    model = CNN().to(device)
    print(model)

    # 5. 定义损失函数
    loss_fn = nn.CrossEntropyLoss()

    # 6. 定义优化器
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # 7. 训练和测试
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}\n-------------------------------")

        train_one_epoch(
            train_dataloader,
            model,
            loss_fn,
            optimizer,
            device
        )

        test(
            test_dataloader,
            model,
            loss_fn,
            device
        )

    print("Done!")

    # 8. 保存模型参数
    save_model(model, MODEL_SAVE_PATH)


if __name__ == "__main__":
    main()