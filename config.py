

# config.py

"""
配置文件：统一管理项目中的超参数和路径。

这样做的好处是：
- 不需要在多个文件里反复修改参数
- 训练配置更清晰
- 后续改 batch size、epoch、学习率时更方便
"""

# 数据相关配置
DATA_DIR = "data"
BATCH_SIZE = 64

# 训练相关配置
EPOCHS = 5
LEARNING_RATE = 1e-3

# 模型保存路径
MODEL_SAVE_PATH = "model.pth"

# 分类类别数量
NUM_CLASSES = 10