# coding: utf-8
# Project：ahu_cv
# Author：XHao
# Date ：2023-6-12 17:16
# Tool ：PyCharm
import torch


def backward():
    x = torch.arange(4.0, requires_grad=True)
    y = 2 * torch.dot(x, x)
    y.backward()
    print(f"第一次求导，x的grad：{x.grad}")
    x.grad.zero_()
    y = x * x  # y对于x的倒数，是[y(i)/x(i),]不是[
    #                                [y(i)/x(j),y(i+1)/x(j),...],
    #                                ]
    # y.sum().backward(10)      # sum()将y变成标量,在求导是
    y.backward(torch.tensor([1, 1, 1, 3]))  # 这里y
    print(f"第二次求导，x的grad：{x.grad}")


if __name__ == '__main__':
    backward()
