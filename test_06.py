import torch
import numpy as np
from torch.utils import data
import matplotlib.pyplot as plt
import random
from torch import nn


def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声。"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器。"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    #随机读取相关的样本
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]  # yield是一种迭代器形式的return，每一次是从上一次的位置开始return

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

"""
batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
"""

# plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
# plt.show()

batch_size = 10  # 如果是剩下的最后一个的不够一个batch，那么一般是直接将最后一个的不满的数量作为一个batch的；\
                 # 当然也有直接扔掉的；\
                 # 还有从下一个epoch里面补上相关数量的凑齐(随机采样)
data_iter = load_array((features, labels), batch_size)

net = nn.Sequential(nn.Linear(2, 1))  # 这是类似于容器的，类似于一个存储一堆神经网络的list
net[0].weight.data.normal_(0, 0.01)  # 初始化参数正态分布
net[0].bias.data.fill_(0)  # 权重初始化为0
loss = nn.MSELoss()  # 为什么是均值，batch上面求梯度的话，没有求均值的话会太大；同时如果batch size变化的话，求均值可以保证梯度的大小不会发生太大的变化\
                     # （当然上面的两点都可以通过调整学习率来达到同样的效果）
                     # 学习率应该如何设置：1.选择合适的优化算法，对于学习率没有这么敏感
                     # 2. 初始化参数
                     # 3，老师还会讲调参的部分
trainer = torch.optim.SGD(net.parameters(), lr=0.03)  # 设置相关的loss 和优化器和lr

num_epochs = 3
for epoch in range(num_epochs):  # epoch 一般是人定，不过多训练点没有太大问题
    for X, y in data_iter:
        # 批量大小和学习率
        # 学习率和批量大小理论上不会影响到收敛结果
        # 只要不是特别大，一般来说都是能够收敛的
        # 特别大可不好哦
        l = loss(net(X), y)
        trainer.zero_grad()  # 梯度清零
        l.backward()  # 反向传播，梯度计算到参数w,b
        # 为啥不用牛顿法，深度学习中的损失函数和优化模型严格数学解没有实际意义；牛顿法非凸优化收敛不到最优解（无约束非线性的优化）
        trainer.step()  # 调用step函数进行模型更新
    with torch.no_grad():
        l = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {l:f}')


