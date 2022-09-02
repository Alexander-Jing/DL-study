import torch
from torch import nn
from torch.nn import functional as F

# pytorch中的module概念
# 在pytorch中，任何一个层和神经网络都应该是一个module的子类
# 各种构造网络的方法

class MLP(nn.Module):
    def __int__(self):
        super(MLP, self).__int__()  # 继承nn.Module的变量和方法
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))

class MySequential(nn.Module):
    def __int__(self, *args):  # 传入的是list
        super(MySequential, self).__int__()
        for block in args:
            self._modules[block] = block  # self._modules其实是父类的变量，本质上是一个ordered dict，是一个有序字典

    def forward(self, X):  # 据说这里的forward前向函数没有被重写，和原来的版本是一样的
        for block in self._modules.values():
            X = block(X)
        return X
# net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

class FixedHiddenMLP(nn.Module):
    def __int__(self):
        super(FixedHiddenMLP, self).__int__()
        self.rand_weight = torch.rand((20, 20), requires_grad=False)  # 随机的初始化权重，无需计算梯度, 固定的数值
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        X = F.relu(torch.mm(X, self.rand_weight) + 1)  # 自定义的定义模型的形式
        X = self.linear(X)
        while X.abs().sum() > 1:  # 自己随机设置的操作
            X /= 2
        return X.sum()

class NestMLP(nn.Module):  # 混合形式的构造方法
    def __int__(self):
        super(NestMLP, self).__int__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())  # 灵活构造形式
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())  # 混合形式

## 参数管理
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
print(net[2].state_dict())  # 显示最后一层的参数
print(net[2].bias)  # 这里显示的是parameter参数
print(net[2].bias.date)  # 这里显示的是数值本身
print(net[2].weight.grad)  # 显示梯度gradient
print(*[(name, param.shape) for name , param in net.named_parameters()])  # 显示里面对应层的weights 和 bias 参数
print(net.state_dict()['2.bias'].data)  # 直接显示对应的层的参数

