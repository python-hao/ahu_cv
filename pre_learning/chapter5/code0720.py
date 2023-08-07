# coding: utf-8
# Project：ahu_cv
# Author：XHao
# Date ：2023-7-20 10:07
# Tool ：PyCharm
import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, X):
        h1 = self.hidden(X)
        return self.out(F.relu(h1))


if __name__ == '__main__':

    net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
    # print(net[0].weight)  # 尚未初始化
    print(net)

    X = torch.rand(2, 20)
    net(X)
    print(net)
