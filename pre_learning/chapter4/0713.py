# coding: utf-8
# Project：ahu_cv
# Author：XHao
# Date ：2023-7-13 14:01
# Tool ：PyCharm`
import threading

import torch
from torch import nn

from pre_learning.xhaoTools.plot import XhaoPlot, AnimatorPlot
from pre_learning.xhaoTools import d2l


# x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
# y = torch.relu(x)
#
# plot = XhaoPlot()
# ax_relu = plot.subplot(211)
# plot.plot(x.detach(), y.detach(), xlabel='x', ylabel='relu',axes=ax_relu, figsize=(5, 2.5))
#
# y.backward(torch.ones_like(x), retain_graph=True)
# ax_grad = plot.subplot(212)
# plot.plot(x.detach(), x.grad, xlabel='x', ylabel='grad of relu', axes=ax_grad, figsize=(5, 2.5))
# plot.show()

def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)


def MLP_detail():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    num_inputs, num_outputs, num_hiddens = 784, 10, 256
    W1 = nn.Parameter(torch.randn(
        num_inputs, num_hiddens, requires_grad=True) * 0.01)
    b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
    W2 = nn.Parameter(torch.randn(
        num_hiddens, num_outputs, requires_grad=True) * 0.01)
    b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
    params = [W1, b1, W2, b2]

    def net(X):
        X = X.reshape((-1, num_inputs))
        H = relu(X @ W1 + b1)  # 这⾥“@”代表矩阵乘法
        return (H @ W2 + b2)

    loss = nn.CrossEntropyLoss(reduction='none')  # reduction='none'表示直接返回n个样本的loss，张量
    num_epochs, lr = 10, 0.1
    updater = torch.optim.SGD(params, lr=lr)

    animator = AnimatorPlot(frames=10, interval=1)
    animator.update([], [[], [], []])
    animator.set_axes(legend=['train loss', 'train acc', 'test acc'], xlim=(0, num_epochs), ylim=(0, 1))
    # train_ch3(net, train_iter, test_iter, cross_entropy, num_epoch, torch.optim.SGD([W, b], lr), animator)
    th = threading.Thread(target=d2l.train_ch3, args=(
        net, train_iter, test_iter, loss, num_epochs, updater, device, animator))
    th.start()
    animator.show()


def MLP_fast():

    def init_weight(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(784, 256),
                        nn.ReLU(),
                        nn.Linear(256, 10))

    net.apply(init_weight)
    net.to(device)

    batch_size, lr, num_epochs = 256, 0.1, 10
    loss = nn.CrossEntropyLoss(reduction='none')
    updater = torch.optim.SGD(net.parameters(), lr=lr)
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    animator = AnimatorPlot(frames=10, interval=1)
    animator.update([], [[], [], []])
    animator.set_axes(legend=['train loss', 'train acc', 'test acc'], xlim=(0, num_epochs), ylim=(0, 1))
    th = threading.Thread(target=d2l.train_ch3, args=(
        net, train_iter, test_iter, loss, num_epochs, updater, device))
    th.start()
    animator.show()


if __name__ == '__main__':
    MLP_fast()
