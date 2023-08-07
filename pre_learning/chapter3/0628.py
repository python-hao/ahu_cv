# coding: utf-8
# Project：ahu_cv
# Author：XHao
# Date ：2023-6-28 09:36
# Tool ：PyCharm

import os
import threading
import time

import torch
import torchvision
from torch import nn
from torch.utils import data
from torchvision import transforms
from matplotlib import pyplot as plt

from pre_learning.xhaoTools import d2l
from pre_learning.xhaoTools.plot import AnimatorPlot, XhaoPlot


def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的⽂本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def get_dataloader_workers():
    """使⽤4个进程来读取数据"""
    return 0

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 这⾥应⽤了⼴播机制


def net(X_):
    return softmax(torch.matmul(X_.reshape((-1, W.shape[0])), W) + b)


def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])



def softmax_detail():
    # print(len(mnist_train), len(mnist_test))
    # X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
    # show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, device)
    # # 测试 读取训练数据的速度
    # start = time.time()
    # for X, y in iter(train_iter):
    #     continue
    # print(f"{time.time()- start:.2f} sec")

    # X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    # print(X.sum(0, keepdim=True), X.sum(1, keepdim=True))

    X = torch.normal(0, 1, (2, 5))
    # X_prob = softmax(X)
    #
    # y = torch.tensor([0, 2])
    # y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
    # print(evaluate_accuracy(net, test_iter, device))

    lr = 0.1
    num_epoch = 10
    animator = AnimatorPlot(frames=10, interval=1)
    animator.update([], [[], [], []])
    animator.set_axes(legend=['train loss', 'train acc', 'test acc'], xlim=(0, num_epoch), ylim=(0.3, 1))
    # train_ch3(net, train_iter, test_iter, cross_entropy, num_epoch, torch.optim.SGD([W, b], lr), animator)
    th = threading.Thread(target=d2l.train_ch3, args=(
        net, train_iter, test_iter, cross_entropy, num_epoch, torch.optim.SGD([W, b], lr), animator))
    th.start()
    animator.show()


def softmax_fast():
    def init_weight(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)

    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    net.apply(init_weight)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    net.to(device)

    loss = nn.CrossEntropyLoss()

    optim = torch.optim.SGD(net.parameters(), lr=0.1)

    num_epochs = 10
    batch_size = 128
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    animator = AnimatorPlot(frames=10, interval=1)
    animator.update([], [[], [], []])
    animator.set_axes(legend=['train loss', 'train acc', 'test acc'], xlim=(0, num_epochs), ylim=(0, 1))
    # train_ch3(net, train_iter, test_iter, cross_entropy, num_epoch, torch.optim.SGD([W, b], lr), animator)
    th = threading.Thread(target=d2l.train_ch3, args=(
        net, train_iter, test_iter, loss, num_epochs, optim, device, animator))
    th.start()
    animator.show()


if __name__ == '__main__':
    num_inputs = 784  # 28x28=784
    num_outputs = 10
    W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)

    softmax_fast()
