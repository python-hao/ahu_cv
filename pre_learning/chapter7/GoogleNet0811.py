import sys,os
sys.path.append(os.getcwd())
from pathlib import Path
from torch import nn
from torch.nn import functional as F
import torch

class Inception(nn.Moudle):
    def __init__(self, inchannels, c1,c2,c3,c4,**kwargs):
        super(Inception, self).__init__(**kwargs)
        # 线路1，单1x1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1x1卷积层后接5x5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3x3最⼤汇聚层后接1x1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1)))
        p3 = F.relu(self.p2_3(F.relu(self.p3_1)))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)

class BatchNormal(nn.Module):
    def __init__(self,num_features, num_dims, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        self.gama = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.ones(shape))
        self.moving_mean = torch.ones(shape)
        self.moving_var = torch.ones(shape)
    def forward(self, x):
        if self.moving_mean.device != x.device:
            self.moving_mean = self.moving_mean.to(x.device)
            self.moving_var = self.moving_var.to(x.device)
        y, self.moving_mean, self.moving_var = batch_normal(
            x,self.gama, self.beta, self.moving_mean, self.moving_var, eps=1e-5, momentum=0.9
        )
        return y
    

def batch_normal(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 通过is_grad_enabled来判断当前模式是训练模式还是预测模式
    if not torch.is_grad_enabled():
        # 如果是在预测模式下，直接使⽤传⼊的移动平均所得的均值和⽅差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 使⽤全连接层的情况，计算特征维上的均值和⽅差
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 使⽤⼆维卷积层的情况，计算通道维上（axis=1）的均值和⽅差。
            # 这⾥我们需要保持X的形状以便后⾯可以做⼴播运算
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # 训练模式下，⽤当前的均值和⽅差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新移动平均的均值和⽅差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta # 缩放和移位
    return Y, moving_mean.data, moving_var.data

if __name__ == '__main__':
    pass
