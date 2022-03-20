import torch
import numpy as np
import pandas as pd
import argparse
import cv2
import os

"""
according to LIMu 2.1-2.3
https://zh-v2.d2l.ai/chapter_preliminaries/linear-algebra.html
2022.3.20
"""

def tensor_test():
    """
    this is the tensor data structure test

    args:
    returns:
    cautions:
    """
    X = torch.arange(12).reshape((3,4))
    Y = [i for i in range(12)]
    Y = torch.tensor(Y).reshape((3,4))

    before_Y = id(Y)  # check the memory ID of Y
    Y += X
    print(id(Y) == before_Y)
    Y = X + Y
    print(id(Y) == before_Y)  # different kinds of assignment lead to different ways of ID change

def data_test():
    """
    create a dataset in the form of .csv

    args:
    returns:
    cautions:
    """
    # create the dataset
    os.makedirs(os.path.join('..', 'data'), exist_ok=True)
    data_file = os.path.join('..', 'data', 'house_tiny.csv')
    with open(data_file, 'w') as f:
        f.write('NumRooms,Alley,Price\n')  # 列名
        f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
        f.write('2,NA,106000\n')
        f.write('4,NA,178100\n')
        f.write('NA,NA,140000\n')
    # read the data
    data = pd.read_csv(data_file)
    # compensate for the losing data
    inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
    inputs = inputs.fillna(inputs.mean())
    # translate the NaN or Pave to the one-hot-like coding
    inputs = pd.get_dummies(inputs, dummy_na=True)
    X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
    print("translate the pandas data to the form of tensor")

def linear_algebra():
    """
    test for some linear algebra factors in torch
    """
    a = torch.eye(4)
    print(a.cumsum(axis=0))  # the accumulation
    b = torch.eye(4)
    print(torch.dot(a[0,:],b[0,:]))  # 1D tensors expected
    print(torch.sum(a*b))  # torch.dot(a,b) equals torch.sum(a*b), this is different from the np.dot()

    print(torch.mm(a,b))  # matrix multiple in torch
    print(torch.norm(a))  # the L2-norm/F-norm of a vector/matrix
    print(torch.abs(a).sum())  # the L1-norm of a matrix

    a = torch.ones((2,5,4))
    print(a.sum(axis=0).shape)  # shape (5, 4)
    print(a.sum(axis=1).shape)  # shape (2, 4)
    print(a.sum(axis=2).shape)  # shape (2, 5)
    print(a.sum(axis=(1,2)).shape)  # shape (2, )
    print(a.sum(axis=1, keepdim=True).shape)  # shape (2,1,4), it will keep the dimension


linear_algebra()

#data_test()
#tensor_test()