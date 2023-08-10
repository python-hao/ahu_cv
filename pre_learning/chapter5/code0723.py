# coding: utf-8
# Project：ahu_cv
# Author：XHao
# Date ：2023-7-23 01:02
# Tool ：PyCharm

import torch
import torch.nn.functional as F
from torch import nn


class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.zeros(units, ))

    def forward(self, x):
        # y = X*W + bias
        linear = torch.matmul(x, self.weight.data) + self.bias.data
        return F.relu(linear)


def save_load_torch_file():
    model = MyLinear(5, 3)
    torch.save(model.state_dict(), 'model.parameters')

    model2 = MyLinear(5, 3)
    model2.load_state_dict(torch.load('model.parameters'))

    x = torch.randn(6, 5)
    out1 = model(x)
    out2 = model2(x)
    print(out2 == out1)


if __name__ == '__main__':
    # dense = MyLinear(5, 3)
    # out = dense(torch.randn(2, 5))
    # print(out)

    # save_load_torch_file()

    gpu_num = torch.cuda.device_count()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(5, 3, device=device)
    net = nn.Sequential(MyLinear(5, 3))
    net = net.to(device)
    
