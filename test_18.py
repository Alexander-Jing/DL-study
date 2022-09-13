import torch
from torch import nn
from torch.nn import functional as F

# 卷积层
# 计算方式[(n-k+p+s)/s], n:宽度, k:kernel size, p:padding, s:stride
conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)
X = torch.ones((6, 8))
X[:, 2:6] = 0
Y = torch.zeros((6, 7))
X[:, 1] = 1; X[:, -2] = -1

X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))

"""
for i in range(20):
    Y_pre = conv2d(X)
    loss = (Y_pre - Y)**2
    conv2d.zero_grad()
    loss.sum().backward()
    conv2d.weight.data[:] -= 3e-2 * conv2d.weight.data
    if (i + 1) % 2 == 0:
        print(f'batch {i+1}, loss {loss.sum():.3f}')
"""

# 池化层
X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
pool2d = nn.MaxPool2d(3)
# pool2d(X) # 步幅和池化大小相同
X = torch.cat((X, X + 1), 1) # 在1号轴上进行合并
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)