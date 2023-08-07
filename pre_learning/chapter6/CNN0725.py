# coding: utf-8
# Project：ahu_cv
# Author：XHao
# Date ：2023-7-25 09:54
# Tool ：PyCharm
import torch
from torch import nn


def corr2d_multi_channel(K, X):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    # 全连接层中的矩阵乘法
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))


X_ = torch.arange(27, dtype=torch.float32).reshape(3, 3, 3)
K_ = torch.arange(3, dtype=torch.float32).reshape(1, 3, 1, 1)

Y1 = corr2d_multi_channel(K_, X_)
print(Y1)

# class LeNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, x):
#         return x.view(-1, 1, 28, 28)
#
#
# if __name__ == '__main__':
#     net = nn.Sequential(
#
#     )
