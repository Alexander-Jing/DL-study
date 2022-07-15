import torch
import torch.nn as nn
import torchvision
from torch.utils import data
from torchvision import transforms

# drop out
# 对于隐藏层中的某一层的每一个神经元的输出x，有概率p设置成0，还有概率(1-p)设置成x/(1-p)，这样均值计算出来还是E(x)=x，但是确实设置成0的神经元算是舍弃掉了一部分
# drop out也算是一种正则项

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

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1:
        return torch.zeros_like(X)
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape) > dropout).float()  # 这个mask就是用来设置这一层X的权重的，有dropout的概率为0， (1-dropout)的概率为1
    return mask * X / (1.0 - dropout)




class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 is_training = True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # 只有在训练模型时才使用dropout
        if self.training == True:
            # 在第一个全连接层之后添加一个dropout层
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            # 在第二个全连接层之后添加一个dropout层
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out


if __name__ == '__main__':

    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=None)
    dropout1, dropout2 = 0.2, 0.5
    # 定义了一个展平层，用来把原来的高维数据第0维度保留，其他的展开成为一个一维的向量
    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Dropout(dropout1), nn.Linear(256, 10), nn.Softmax(dim=1))  # 多层感知机的模型，添加了dropout，添加了dropout层
    net.apply(init_weights)  # 初始化

    # num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
    # net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
    # net.apply(init_weights)  # 初始化

    loss = nn.CrossEntropyLoss()
    # trainer = torch.optim.SGD(net.parameters(), lr=0.1)
    weight_decay = list()
    bias = list()
    #for m in net:
    #    print(m)
    trainer = torch.optim.SGD([
        {"params": net[1].weight,
         'weight_decay': 0.001},  # 正则项的部分，权重衰退，参数是对应的λ，一般设置为0.001(e^{-3})，不会设置太大到1.0那种
        {"params": net[1].bias},  # 注意这里只对于w权重做约束，偏置w不考虑进来的
        {"params": net[4].weight,
         'weight_decay': 0.001},  # 正则项的部分，权重衰退
        {"params": net[4].bias},
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



