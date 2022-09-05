import torch
from torch import nn
from torch.nn import functional as F

## 选择服务器
torch.device('cpu')
torch.cuda.device('cuda:1')
torch.cuda.device_count()

def try_gpu(i=0):  # 返回第i个gpu，如果没有，则返回cpu()
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():  # 返回所有可用的GPU，如果没有，则返回cpu()
    devices =[torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

X = torch.randn(size=(4, 2), device=try_gpu())


net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())  # 参数转移至0号GPU上面，一般预处理在cpu上快一点，然后再送给GPU，具体视实际情况

net(X)  # X是在0号GPU上面的，多个GPU的话，数据需要在同一个GPU上面做运算

# GPU购买要求
# 显存、计算能力、价格
# 两种GPU：云上面的GPU，消费级GPU（普通人买的）
# 不同的型号，核的数量不一样
# 买新的不买旧的，买最贵的，理论上价格和计算性能是正比的
# 计算性能GFlops，价格Price
# 内存是很贵的事情，显存是很贵的事情
#
