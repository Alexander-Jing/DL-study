import numpy as np
import torch
import os
import cv2

"""
LiMu 2.4
2022.4.3
"""
"""
计算图：无环图，用于表示前向计算的图，将代码分解成操作子
构造计算图：显式构造tf
            隐式构造torch（系统自动记住的）
自动求导（反向传播）：
        前向：存储中间结果
        反向：反过来算，需要中间结果
        复杂度：计算复杂度O(n)
                内存复杂度O(n)，十分耗资源
        更正向累积对比（正过来算）：
                内存复杂度O(1)
                但是计算复杂度O(n)，对于每一个变量都要来一次
                基本不用
"""


def auto_grad():
    x = torch.arange(4.0, requires_grad=True)  # 保存梯度
    y = 2 * torch.dot(x, x)  # 显式的计算图 y=2 x^2
    y.backward()  # 测试显示只能求一次导数
    print(x.grad == 4 * x)
    x.grad.zero_()  # 一般来说，torch会累计梯度，所以要清除之前的值，\
    # 可是什么时候要累计梯度呢，算批量的时候，可以把批量切分，在不同的卡上跑，然后累计梯度，\
    # 或者说在多模态的时候，一个weight在不同的模态之间share的时候，梯度累加也是有好处的，\
    # 事实上多个loss反向传播的时候是累加梯度的
    y = x.sum()
    y.backward()
    print(x.grad)

    x.grad.zero_()
    y = x * x
    y.sum().backward()  # 更多情况下都是标量对于向量求导，毕竟很多loss都是标量
    print(x.grad == 2 * x.sum())

    x.grad.zero_()
    y = x * x  # y还是关于x的函数
    u = y.detach()  # 把y设置成为一个常数，不用算梯度了，这是一个常数了，相当于从计算图里面提出来了
    z = u * x  # z不就是一个常数乘x吗
    z.sum().backward()
    print(x.grad == u)

    x.grad.zero_()
    y.sum().backward()  # 但是关于y对于x求导还是不变的
    print(x.grad == 2 * x)

    # 事实上如果是python的控制流（条件、循环或者任意函数调用，递归应该也可以），仍然可以计算得到变量的梯度
