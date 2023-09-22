# coding: utf-8
# Project：ahu_cv
# Author：XHao
# Date ：2023-7-13 15:41
# Tool ：PyCharm + VsCode

import time

import numpy as np
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from pathlib import Path
from pre_learning.xhaoTools.accumulator import Accumulator


def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的⽂本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def load_array(data_arrays, batch_size, is_train=True):
    """构造⼀个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def load_data_fashion_mnist(batch_size, resize=None):
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    # os.makedirs('data', exist_ok=True)
    path = Path('pre_learning/data')
    path.mkdir(parents=True, exist_ok=True)
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root=path,
        train=True,
        transform=trans,
        download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root=path,
        train=False,
        transform=trans,
        download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=0),
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=0))


def train_epoch_ch3(net, train_iter, loss, updater):
    """训练模型⼀个迭代周期（定义⻅第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
        device = next(iter(net.parameters())).device
        print(device)
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    batch_num = len(train_iter)
    for index, (X, y) in enumerate(train_iter):
        # 计算梯度并更新参数
        print(f'\r--> {index + 1} / {batch_num}', end='')
        X = X.to(device)
        y = y.to(device)
        y_hat = net(X)
        y_hat=y_hat.to(device)
        loss_value = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使⽤PyTorch内置的优化器和损失函数
            updater.zero_grad()
            loss_value.mean().backward()
            updater.step()
        else:
            # 使⽤定制的优化器和损失函数
            loss_value.sum().backward()
            updater(X.shape[0])
        metric.add(float(loss_value.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    print()
    return metric[0] / batch_num, metric[1] / metric[2]


def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    device = next(iter(net.parameters())).device
    with torch.no_grad():
        for X_, y_ in data_iter:
            X_ = X_.to(device)
            y_ = y_.to(device)
            metric.add(accuracy(net(X_), y_), y_.numel())
    return metric[0] / metric[1]


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater, plot=None):
    """训练模型（定义⻅第3章）"""
    x = []
    train_loss = []
    train_acc = []
    test_acc = []
    for epoch in range(num_epochs):
        print(f'epoch:{epoch + 1}')
        start_time = time.time()
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc_cur = evaluate_accuracy(net, test_iter)
        print(f'-->用时：{time.time() - start_time} s')
        x.append(epoch)
        train_loss.append(train_metrics[0])
        train_acc.append(train_metrics[1])
        test_acc.append(test_acc_cur)
        # animator.add(epoch + 1, train_metrics +
        if plot:
            plot.update([x], [train_loss, train_acc, test_acc])
        print(f'-->train_loss:{train_metrics[0]}, train_acc:{train_metrics[1]}, test_acc:{test_acc_cur}')
    # train_loss, train_acc = train_metrics

def train_batch_on_multi_device(net, train_iter, test_iter, loss, num_epochs, updater,device_params, plot=None):
    pass

def test():
    path = Path('pre_learning/data').resolve()
    print(path)


class Timer:
    """记录多次运行时间。"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器。"""
        self.tik = time.time()

    def record(self):
        """停止计时器并将时间记录在列表中。"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间。"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和。"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间。"""
        return np.array(self.times).cumsum().tolist()


class Benchmark:
    """⽤于测量运⾏时间"""
    def __init__(self, description='Done'):
        self.description = description

    def __enter__(self):
        self.timer = Timer()
        return self
    
    def __exit__(self, *args):
        print(f'{self.description}: {self.timer.record():.8f} sec')
