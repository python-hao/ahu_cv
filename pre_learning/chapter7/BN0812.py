import torch
from torch import nn
bn = nn.BatchNorm2d(32) # 使用BN层需要传入一个参数num_features，即特征的通道数
print('BN:',bn)

input = torch.randn(4, 32, 224, 224)
output = bn(input)# BN层不改变输入层的特征大小，只改变数值，
print('Output Shape:',output.shape)
