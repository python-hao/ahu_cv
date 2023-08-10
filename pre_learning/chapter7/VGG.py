import sys,os
sys.path.append(os.getcwd())
from pathlib import Path
from torch import nn
import torch

def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        ))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
    
    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(),
        nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(),
        nn.Dropout(0.5), nn.Linear(4096, 10)
    )

def test_shape():
    X = torch.randn(1,1,224,224)
    conv_arch = [(1, 64), (1, 128), (2, 256), (2, 512), (2, 512)]
    net  = vgg(conv_arch)
    for block in net:
        X= block(X)
        print(block.__class__.__name__, "out shape:\t", X.shape)

if __name__ == '__main__':
    test_shape()
    # 参数大小 = kernal 的大小 （长*宽） * kernal的通道（输入的通道）* 输出的通道数