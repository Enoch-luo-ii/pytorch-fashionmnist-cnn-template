# PyTorch MNIST / FashionMNIST 训练模板

## 项目简介

这是一个 PyTorch 训练流程模板项目，用于学习如何把单文件训练代码整理成一个可复用的项目结构。

本项目使用 FashionMNIST 数据集，完成一个 10 分类图片识别任务。

---

## 项目目标

通过这个项目，你需要掌握：

- 如何用 PyTorch 加载数据集
- 如何使用 Dataset 和 DataLoader
- 如何定义 CNN 模型
- 如何写完整训练循环
- 如何在测试集上评估模型
- 如何保存模型参数
- 如何把代码拆成多个文件管理

---

## 项目结构

```text
pytorch_mnist_template/
├── train.py
├── model.py
├── dataset.py
├── utils.py
├── config.py
├── requirements.txt
└── README.md
```

---

## 每个文件的作用

### `config.py`

统一管理项目配置，例如：

- batch size
- epoch 数
- learning rate
- 数据保存路径
- 模型保存路径
- 分类类别数量

### `dataset.py`

负责数据集相关操作：

- 下载 FashionMNIST 数据集
- 创建训练集 DataLoader
- 创建测试集 DataLoader
- 打印 batch 的形状

### `model.py`

定义 CNN 神经网络模型。

模型大致结构：

```text
Conv2d → MaxPool2d → ReLU
Conv2d → MaxPool2d → ReLU
Flatten
Linear
```

### `utils.py`

存放工具函数，例如：

- 自动选择 device
- 保存模型参数
- 加载模型参数

### `train.py`

项目的主入口，负责完整训练流程：

```text
加载数据
↓
创建模型
↓
定义损失函数
↓
定义优化器
↓
训练模型
↓
测试模型
↓
保存模型
```

---

## 安装依赖

在项目根目录下运行：

```bash
pip install -r requirements.txt
```

如果你的文件目前叫 `requirements.py`，建议先把它改名为：

```text
requirements.txt
```

---

## 运行项目

在项目根目录下运行：

```bash
python train.py
```

或者在 PyCharm 中右键 `train.py`，选择：

```text
Run 'train'
```

---

## 核心训练代码

```python
pred = model(X)
loss = loss_fn(pred, y)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

这 5 行代码的含义是：

```text
预测 → 计算 loss → 清空旧梯度 → 反向传播 → 更新参数
```

---

## 模型保存

训练结束后，模型参数会保存到：

```text
model.pth
```

保存代码：

```python
torch.save(model.state_dict(), "model.pth")
```

注意：这里保存的是模型参数，不是完整模型结构。

---

## 运行成功后，你会看到类似输出

```text
Using mps device
Shape of X [N, C, H, W]: torch.Size([64, 1, 28, 28])
Shape of y: torch.Size([64]) torch.int64
CNN(
  ...
)
Epoch 1
-------------------------------
loss: 2.305123  [    0/60000]
loss: 1.982345  [ 6400/60000]
...
Test Error:
Accuracy: 80.0%, Avg loss: 0.543210
```
## 训练结果

训练 5 个 epoch 后，测试集结果：

```text
Accuracy: 69.1%
Avg loss: 0.840870
---

## Day 8 学习总结

Day 8 的重点不是学习新的复杂模型，而是学会工程化组织 PyTorch 代码。

以前我们可能把所有代码都写在一个文件里：

```text
数据加载 + 模型定义 + 训练 + 测试 + 保存模型
```

现在把它拆成：

```text
dataset.py：负责数据
model.py：负责模型
train.py：负责训练
utils.py：负责工具函数
config.py：负责配置
```

这样代码更清晰，也更适合以后做更大的项目。


