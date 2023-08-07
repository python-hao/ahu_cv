# coding: utf-8
# Project：ahu_cv
# Author：XHao
# Date ：2023-7-18 9:34
# Tool ：PyCharm
import hashlib
import os
import tarfile
import threading
import zipfile

import numpy as np
import pandas as pd
import requests
import torch
from torch import nn
from pre_learning.xhaoTools import d2l
from pre_learning.xhaoTools.plot import AnimatorPlot, XhaoPlot


def download(name, DATA_HUB, cache_dir=os.path.join('..', 'data')):
    """下载⼀个DATA_HUB中的⽂件，返回本地⽂件名"""
    assert name in DATA_HUB, f"{name} 不存在于{DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # 命中缓存
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname


def download_extract(name, DATA_HUB, folder=None):
    """下载并解压zip/tar⽂件"""
    fname = download(name, DATA_HUB)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar⽂件可以被解压缩'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir


def download_all(DATA_HUB):
    """下载DATA_HUB中的所有⽂件"""
    for name in DATA_HUB:
        download(name, DATA_HUB)


def get_format_data():
    DATA_HUB = dict()
    DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
    DATA_HUB['kaggle_house_train'] = (
        DATA_URL + 'kaggle_house_pred_train.csv',
        '585e9cc93e70b39160e7921475f9bcd7d31219ce')
    DATA_HUB['kaggle_house_test'] = (
        DATA_URL + 'kaggle_house_pred_test.csv',
        'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')
    train_data = pd.read_csv(download('kaggle_house_train', DATA_HUB))
    test_data = pd.read_csv(download('kaggle_house_test', DATA_HUB))
    all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:-1]))
    # all_features.to_csv(path_or_buf='all.csv')
    # 跳出数字类型的列：int 、 float
    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    # 数字类型的 列 归一化
    all_features[numeric_features] = all_features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std()), axis=0
    )
    # print(all_features[numeric_features].columns[np.where(np.isnan(all_features[numeric_features]))[1]])
    # 填补 缺失的值nan 为均值 0
    all_features[numeric_features] = all_features[numeric_features].fillna(0)
    all_features = pd.get_dummies(all_features, dummy_na=True, dtype=np.float32)
    train_n = train_data.shape[0]
    # pandas 数据转化为 tensor
    train_features = torch.tensor(all_features[:train_n].values, dtype=torch.float32)
    test_features = torch.tensor(all_features[train_n:].values, dtype=torch.float32)
    train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)
    return train_features, train_labels, test_features, test_data


def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


def train(net, train_features, train_labels, valid_features, valid_labels,
          epochs, lr, batch_size, lossFunction, plot):
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    train_loss = []
    valid_loss = []
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    for epoch in range(epochs):
        for index, (x, y) in enumerate(train_iter):
            optim.zero_grad()
            y_ = net(x)
            loss = lossFunction(y_, y)
            loss.backward()
            optim.step()

        clipped_preds = torch.clamp(net(train_features), 1, float('inf'))
        rmse = torch.sqrt(lossFunction(torch.log(clipped_preds),
                                       torch.log(train_labels)))
        train_loss.append(rmse.item())

        if valid_labels is not None:
            clipped_preds = torch.clamp(net(valid_features), 1, float('inf'))
            rmse = torch.sqrt(lossFunction(torch.log(clipped_preds),
                                           torch.log(valid_labels)))
            valid_loss.append(rmse.item())
        if plot:
            plot.update(np.arange(1, epochs + 1), [train_loss, valid_loss])
    return train_loss, valid_loss


def init_weight(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


def run_k_train(k=1, plot=None):
    # 获取数据，并预处理（分成训练和测试），返回tensor类型
    train_features, train_labels, test_features, test_data = get_format_data()

    # 预测房价的值，线性回归，定义普通的损失函数
    lossFunction = nn.MSELoss()

    # 定义 线性回归网络,并初始化权重
    net = nn.Sequential(
        nn.Linear(train_features.shape[1], 1)
    )
    net.apply(init_weight)

    # 定义超参数
    epochs = 100
    batch_size = 64
    lr = 5

    if k <= 1:
        # 普通训练模式
        train_loss, valid_loss = train(net, train_features, train_labels, None, None,
                                       epochs, lr, batch_size, lossFunction, plot)
        preds = net(test_features).detach().numpy()
        # 将其重新格式化以导出到Kaggle
        test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
        submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
        submission.to_csv('submission.csv', index=False)
        torch.save(net.state_dict(), 'net.pth')
        return
    # 进行k折交叉验证的训练
    train_loss_k = []
    valid_loss_k = []
    for i in range(k):
        # 训练数据分为k份，其中一份作为验证，重复k次，使得每一份数据都做过验证集
        train_features_k, train_labels_k, valid_features, valid_labels = get_k_fold_data(k, i, train_features,
                                                                                         train_labels)
        # 每一折训练epochs轮
        train_loss, valid_loss = train(net, train_features_k, train_labels_k, valid_features, valid_labels,
                                       epochs, lr, batch_size, lossFunction, plot)

        # print(f'第{i + 1}折： epoch {epoch + 1}, train_loss {train_loss[-1]:f}, valid_loss {valid_loss[-1]:f}')
        train_loss_k.append(train_loss)
        valid_loss_k.append(valid_loss)
    torch.save(net.state_dict(), 'net.pth')


if __name__ == '__main__':
    xplt = AnimatorPlot(frames=30, interval=1)
    xplt.set_axes(xlabel='epoch', ylabel='rmse', legend=['train', 'valid'], xlim=(1, 100), ylim=(0, 5),
                  xscale='linear', yscale='linear')
    th = threading.Thread(target=run_k_train, args=(1, xplt,))
    th.start()
    xplt.show()
