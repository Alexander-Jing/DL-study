import torch
import torch.nn as nn
import torchvision
from torch.utils import data
from torchvision import transforms

# weight decay
# 实际上就是在损失函数里面加上了一个二范数形式的正则项(L2,对于其中的参数w，是一个 1/2*λ*Sum(w^2) 的形式，但是这里只约束参数，不约束偏置 )，
# 从而达到约束的效果（损失函数最小化，约束项最小化，一个折中的优化结果），
# 注意的是，如果使用正则项，梯度下降求出来的结果会在参数前面乘上一个小于1的权重，一次达到权重衰退的结果，是一个(1-ηλ)*w的形式，其中η是学习率，λ是L2的超参数
# 因为(1-ηλ)是小于1的，所以是权重衰退了
# 这样可以增强模型的泛化能力，降低模型的过拟合性质

def get_dataloader_workers():
    """使用4个进程来读取数据"""
    return 1

def load_data_fashion_mnist(batch_size, resize=None):

    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)

    mnist_train = torchvision.datasets.FashionMNIST(root="./data", train=True,
                                                    transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="./data", train=False,
                                                   transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=get_dataloader_workers()))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)  # 初始化权重


class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def accuracy(y_hat, y):  #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式，这里默认不需要开启梯度，不过不设置好像也没关系
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

if __name__ == '__main__':

    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=None)
    # 定义了一个展平层，用来把原来的高维数据第0维度保留，其他的展开成为一个一维的向量
    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10), nn.Softmax(dim=1))  # 多层感知机的模型
    net.apply(init_weights)  # 初始化
    loss = nn.CrossEntropyLoss()
    # trainer = torch.optim.SGD(net.parameters(), lr=0.1)
    weight_decay = list()
    bias = list()
    #for m in net:
    #    print(m)
    trainer = torch.optim.SGD([
        {"params": net[1].weight,
         'weight_decay': 1.0},  # 正则项的部分，权重衰退
        {"params": net[1].bias},  # 注意这里只对于w权重做约束，偏置w不考虑进来的
        {"params": net[3].weight,
         'weight_decay': 1.0},  # 正则项的部分，权重衰退
        {"params": net[3].bias},
    ], lr=0.001 )
    num_epochs = 3
    for epoch in range(num_epochs):
        net.train()
        metric = Accumulator(3)
        for X, y in train_iter:
            y_pre = net(X)
            l = loss(y_pre, y)
            trainer.zero_grad()
            l.mean().backward()
            trainer.step()
            metric.add(float(l.sum()), accuracy(y_pre, y), y.numel())
        test_acc = evaluate_accuracy(net, test_iter)
        print("epoch:", epoch, " test_acc: ", test_acc)
    train_loss, train_acc = metric[0] / metric[2], metric[1] / metric[2]
    print("train loss:", train_loss, ", train_acc: ", train_acc)



