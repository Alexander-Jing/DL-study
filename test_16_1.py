import torch
from torch import nn
from torch.nn import functional as F

# pytorch中的module概念
# 在pytorch中，任何一个层和神经网络都应该是一个module的子类
# 构造层的方法

class CenteredLayer(nn.Module):
    def __int__(self):
        super(CenteredLayer, self).__int__()

    def forward(self, X):
        return  X - X.mean()

layer = CenteredLayer()
layer(torch.FloatTensor([1, 2, 3, 4, 5]))

## 自己写了一个nn.Linear() 自带参数
class MyLinear(nn.Module):
    def __int__(self, in_units, units):
        super(MyLinear, self).__int__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)

dense = MyLinear(5, 3)

net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))

# 加载和保存
x = torch.arange(4)
torch.save(x, 'x-file')

x2 = torch.load("x-file")

y = torch.zeros(4)
torch.save([x, y], 'x-files')
x2, y2 = torch.load('x-files')

mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')



X = torch.arange(20)
class MLP(nn.Module):
    def __int__(self):
        super(MLP, self).__int__()  # 继承nn.Module的变量和方法
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)  # module里面直接将net(X)等价为net.forward(X)设计了，所以直接写net(X)就行了
torch.save(net.state_dict(), 'mlp.params')  # 仅仅是保存权重

clone = MLP()  # 需要知道计算图结构
clone.load_state_dict(torch.load("mlp.params"))
clone.eval()


Y_clone = clone(X)
print(Y_clone == Y)

## 选择服务器
torch.device('cpu')
torch.cuda.device('cuda:1')
torch.cuda.device_count()