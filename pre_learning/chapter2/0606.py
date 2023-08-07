# coding: utf-8
# Project：learning
# Author：XHao
# Date ：2023-6-6 15:58
# Tool ：PyCharm

import torch

# # 原地操作
# x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
# y = torch.tensor([2, 2, 2, 2, 2, 6]).reshape((-1, 3))
# before = id(x)
# x = x + y
# x += y
# print(x, before==id(x))

# shape不不一样的，自动广播至shape的最小公倍数，注意的是：每个维度互相成整数倍
a = torch.arange(8).reshape((8, 1))
b = torch.arange(2).reshape((1, 2))
# a = torch.arange(8).reshape((4, 2))
# b = torch.arange(2).reshape((1, 2))
# # 以下均报错：dimension 1 不匹配
# a = torch.arange(8).reshape((1, 8))
# b = torch.arange(2).reshape((1, 2))
# a = torch.arange(8).reshape((2, 4))
# b = torch.arange(2).reshape((1, 2))
# print(a)

# # 内存变化   tensor 和ndarray互化
# numpy_a = a.numpy()
# print(id(numpy_a) == id(a))
# tensor_a = torch.tensor(numpy_a)
# print(id(numpy_a) == id(tensor_a))
# # 内存变化   赋值
# before = id(a)
# a = a + b
# after1 = id(a)
# a += b
# after2 =id(a)
# print(before==after1, after1==after2)


# import os
#
# os.makedirs(os.path.join('..', 'data'), exist_ok=True)  # 第二个参数为True，表示如果文件夹存在则忽略，不存在则创建
# data_file = os.path.join('..', 'data', 'house_tiny.csv')
# with open(data_file, 'w') as f:
#     f.write('NumRooms,Alley,Price\n')  # 列名
#     f.write('NA,Pave,127500\n')  # 每⾏表⽰⼀个数据样本
#     f.write('2,NA,106000\n')
#     f.write('4,NA,178100\n')
#     f.write('NA,NA,140000\n')
#
# # 读取csv文件
# import pandas as pd
#
# data = pd.read_csv(data_file)
# # print(data)
#
# inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
# inputs = inputs.fillna(inputs.mean())
# inputs = pd.get_dummies(inputs, dummy_na=True)


# # 找出缺失最多的列
# nan_num = inputs.isnull().sum(axis=0)
# nan_max_id = nan_num.idxmax()
# # 删除列
# inputs = inputs.drop([nan_max_id], axis=1)
# print(inputs)
