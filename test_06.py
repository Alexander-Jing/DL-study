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

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

net = nn.Sequential(nn.Linear(2,1))  # 这是类似于容器的，类似于一个存储一堆神经网络的list
net[0].weight.data.normal_(0, 0.01)  # 初始化参数正态分布
net[0].bias.data.fill_(0)  # 权重初始化为0
loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.03)  # 设置相关的loss 和优化器和lr

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()  # 梯度清零
        l.backward()  # 反向传播，梯度计算到参数w,b
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')


