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

