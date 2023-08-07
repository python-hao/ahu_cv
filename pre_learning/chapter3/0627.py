# coding: utf-8
# Project：ahu_cv
# Author：XHao
# Date ：2023-6-27 13:31
# Tool ：PyCharm

# 利用pytorch的库来实现线性回归

import numpy as np
import torch
from torch.utils import data
from torch import nn


def synthetic_data(w, b, num_examples):
    """⽣成y=xw+b+噪声"""
    # 从正态分布（0，1）中随机选择样本，样本shape为（num_examples， len（w））
    x = torch.normal(0, 1, (num_examples, len(w)))
    # 计算 y=xw+b
    y = torch.matmul(x, w) + b
    emsenta = torch.normal(0, 0.01, y.shape)
    y += emsenta
    return x, y.reshape((-1, 1))


def load_array(data_arrays, batch_size, is_train=True):
    """构造⼀个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

batch_size = 10
data_iter = load_array((features, labels), batch_size)
# print(next(iter(data_iter)))
net = nn.Sequential(nn.Linear(2, 1))
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

lossFunction = nn.MSELoss()
# lossFunction = nn.HuberLoss()
optim = torch.optim.SGD(net.parameters(), lr=0.03)

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        loss = lossFunction(net(X), y)
        optim.zero_grad()
        loss.backward()
        optim.step()
    loss = lossFunction(net(features), labels)
    print(f'epoch {epoch + 1}, loss {loss:f}')

# 查看网络参数
for name, parameters in net.named_parameters():
    print(name, ':', parameters.grad)
    print(name, ':', parameters.data)
