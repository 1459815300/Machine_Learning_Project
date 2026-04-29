import os
import time

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd


def create_dataset():
    """
    读取手机价格数据集，完成数据划分、标准化和 TensorDataset 封装。

    返回：
        train_dataset: 训练集数据
        valid_dataset: 验证集数据
        input_dim: 输入特征维度
        output_dim: 输出类别数量
    """

    # 1. 读取 CSV 数据文件
    # 假设最后一列是标签，其余列都是特征
    data = pd.read_csv('./data/手机价格预测.csv')

    # 2. 分离特征 x 和标签 y
    # x：所有行，除最后一列之外的所有列
    # y：所有行，最后一列
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # 3. 转换数据类型
    # 神经网络输入一般使用 float32
    # CrossEntropyLoss 要求分类标签是 long/int64 类型
    x = x.astype('float32')
    y = y.astype('int64')

    # 4. 划分训练集和验证集
    # test_size=0.2 表示 80% 训练，20% 验证
    # random_state 固定随机种子，保证每次划分结果一致
    # stratify=y 保持训练集和验证集中各类别比例基本一致
    x_train, x_valid, y_train, y_valid = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=7,
        stratify=y
    )

    # 5. 对特征进行标准化
    # StandardScaler 会将数据转换为均值为 0、标准差为 1 的分布
    # 注意：
    # 训练集用 fit_transform
    # 验证集只能用 transform，不能重新 fit
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_valid = scaler.transform(x_valid)

    # 6. 将 numpy 数组转换为 PyTorch Tensor
    # TensorDataset 可以把特征和标签封装成 PyTorch 数据集
    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train.values, dtype=torch.long)
    )

    valid_dataset = TensorDataset(
        torch.tensor(x_valid, dtype=torch.float32),
        torch.tensor(y_valid.values, dtype=torch.long)
    )

    # 7. 获取输入维度和输出类别数
    # input_dim：每个样本有多少个特征
    # output_dim：分类任务有多少个类别
    input_dim = x_train.shape[1]
    output_dim = len(np.unique(y))

    return train_dataset, valid_dataset, input_dim, output_dim


class ANNModel(nn.Module):
    """
    人工神经网络模型。

    该模型用于多分类任务：
        输入层：input_dim
        隐藏层：128 -> 256 -> 512 -> 128
        输出层：output_dim

    注意：
        最后一层不需要加 Softmax。
        因为 nn.CrossEntropyLoss() 内部已经包含了 LogSoftmax。
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()

        # 使用 nn.Sequential 顺序搭建网络结构
        self.model = nn.Sequential(
            # 第一层全连接层：input_dim -> 128
            nn.Linear(input_dim, 128),
            nn.ReLU(),

            # 第二层全连接层：128 -> 256
            nn.Linear(128, 256),
            nn.ReLU(),

            # 第三层全连接层：256 -> 512
            nn.Linear(256, 512),
            nn.ReLU(),

            # 第四层全连接层：512 -> 128
            nn.Linear(512, 128),
            nn.ReLU(),

            # 输出层：128 -> output_dim
            # output_dim 对应分类类别数量
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        """
        前向传播函数。

        参数：
            x: 输入特征张量，形状为 [batch_size, input_dim]

        返回：
            输出 logits，形状为 [batch_size, output_dim]
        """

        return self.model(x)


def train(train_dataset, input_dim, output_dim):
    """
    训练 ANN 模型。

    参数：
        train_dataset: 训练数据集
        input_dim: 输入特征维度
        output_dim: 输出类别数量
    """

    # 固定 PyTorch 随机种子，使实验结果更容易复现
    torch.manual_seed(10)

    # 使用 DataLoader 按 batch 加载训练数据
    # batch_size=16 表示每次取 16 个样本训练
    # shuffle=True 表示每轮训练前打乱训练数据
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True
    )

    # 创建模型对象
    model = ANNModel(input_dim, output_dim)

    # 多分类任务使用交叉熵损失函数
    # CrossEntropyLoss 的输入是 logits，不需要提前做 Softmax
    criterion = nn.CrossEntropyLoss()

    # 使用 Adam 优化器更新模型参数
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 设置训练轮数
    epochs = 50

    # 开始训练
    for epoch in range(epochs):
        # 设置模型为训练模式
        # 对 Dropout、BatchNorm 等层会有影响
        model.train()

        # 记录当前 epoch 开始时间
        start = time.time()

        # 用于累计当前 epoch 的总损失
        total_loss = 0.0

        # 记录当前 epoch 中 batch 的数量
        batch_num = 0

        # 按 batch 取出训练数据
        for x, y in train_loader:
            # 1. 前向传播，得到模型预测结果
            y_pred = model(x)

            # 2. 计算损失
            loss = criterion(y_pred, y)

            # 3. 清空上一轮的梯度
            optimizer.zero_grad()

            # 4. 反向传播，计算当前梯度
            loss.backward()

            # 5. 根据梯度更新模型参数
            optimizer.step()

            # 6. 累计 loss
            total_loss += loss.item()
            batch_num += 1

        # 计算当前 epoch 的平均损失
        avg_loss = total_loss / batch_num

        # 打印训练过程信息
        print(
            f'Epoch {epoch + 1:02d} | '
            f'Loss {avg_loss:.4f} | '
            f'Time {time.time() - start:.2f}s'
        )

    # 如果 model 文件夹不存在，则自动创建
    os.makedirs('./model', exist_ok=True)

    # 保存模型参数
    # 注意这里只保存 state_dict，不保存整个模型对象
    torch.save(model.state_dict(), './model/phone.pth')

    print('模型已保存到 ./model/phone.pth')


def evaluate(valid_dataset, input_dim, output_dim):
    """
    在验证集上评估模型准确率。

    参数：
        valid_dataset: 验证数据集
        input_dim: 输入特征维度
        output_dim: 输出类别数量
    """

    # 创建和训练时结构完全相同的模型
    model = ANNModel(input_dim, output_dim)

    # 加载训练好的模型参数
    # map_location='cpu' 表示无论模型在哪个设备训练，都加载到 CPU
    model.load_state_dict(torch.load('./model/phone.pth', map_location='cpu'))

    # 使用 DataLoader 加载验证数据
    # 验证集不需要打乱，所以 shuffle=False
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=16,
        shuffle=False
    )

    # 设置模型为评估模式
    model.eval()

    # correct 记录预测正确的样本数
    correct = 0

    # total 记录总样本数
    total = 0

    # 评估阶段不需要计算梯度
    # 使用 torch.no_grad() 可以节省内存和计算资源
    with torch.no_grad():
        for x, y in valid_loader:
            # 1. 前向传播，得到每个类别的 logits
            outputs = model(x)

            # 2. 取 logits 最大的位置作为预测类别
            # dim=1 表示在类别维度上取最大值
            pred = torch.argmax(outputs, dim=1)

            # 3. 统计预测正确的样本数量
            correct += (pred == y).sum().item()

            # 4. 统计样本总数
            total += y.size(0)

    # 计算准确率
    acc = correct / total

    # 输出准确率
    print(f'Accuracy: {acc * 100:.2f}%')


if __name__ == '__main__':
    """
    程序入口。

    执行流程：
        1. 创建数据集
        2. 打印输入维度和类别数
        3. 训练模型
        4. 评估模型
    """

    # 创建训练集、验证集，并获取输入维度和输出类别数
    train_dataset, valid_dataset, input_dim, output_dim = create_dataset()

    # 打印数据基本信息
    print('输入特征维度:', input_dim)
    print('输出类别数量:', output_dim)

    # 训练模型
    train(train_dataset, input_dim, output_dim)

    # 评估模型
    evaluate(valid_dataset, input_dim, output_dim)