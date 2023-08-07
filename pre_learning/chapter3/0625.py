# coding: utf-8
# Project：ahu_cv
# Author：XHao
# Date ：2023-6-25 17:26
# Tool ：PyCharm
import random
import time

import torch


def synthetic_data(w, b, num_examples):  # @save
    """⽣成y=xw+b+噪声"""
    # 从正态分布（0，1）中随机选择样本，样本shape为（num_examples， len（w））
    x = torch.normal(0, 1, (num_examples, len(w)))
    # 计算 y=xw+b
    y = torch.matmul(x, w) + b
    emsenta = torch.normal(0, 0.01, y.shape)
    y += emsenta
    return x, y.reshape((-1, 1))


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)


# 画图--数据集
# from pre_learning.chapter2.xhao_plot import XhaoPlot
# plt = XhaoPlot()
# axes, line = plt.scatter(features[:, 1].detach().numpy(), [labels.detach().numpy(), ],
#                          s=10, alpha=0.5)
# plt.show()

# 构造生成器读取数据集，生成器节省内存，靠yield关键字实现；
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    # 这些样本是随机读取的，没有特定的顺序,为了保证所有的样本数都奔选取，可以利用索引实现：
    # 将索引打乱顺序，这样就生成一个乱序索引列表，顺序访问完这个乱序索引列表就能达到目的
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


# batch_size = 10
# dataloader = data_iter(batch_size, features, labels)
# X, y = next(dataloader)
# print(X, '\n', y)
# X, y = next(dataloader)
# print(X, '\n', y)


# 下面开始一个深度学习的完整步骤：
# 随机初试化参数
# w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
w = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


# 定义模型
def linreg(X, w, b):
    """简单线性回归的模型（一个神经元）"""
    return torch.matmul(X, w) + b


# 定义损失函数
def squared_loss(y_hat, y):
    """均⽅损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# 定义优化算法---如何更新权重参数
def sgd(params, lr, batch_size):
    """⼩批量随机梯度下降"""
    # 该模块下新产生的所有变量都不求导，张量的requires_grad都自动设置为False
    with torch.no_grad():
        for param in params:
            # 参数更新
            param -= lr * param.grad / batch_size
            # 梯度归零
            param.grad.zero_()


lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss
batch_size = 10

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # X和y的⼩批量损失
        # 因为l形状是(batch_size,1)，⽽不是⼀个标量。l中的所有元素被加到⼀起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使⽤参数的梯度更新参数
    # 计算最终结果的Loss
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
    print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
