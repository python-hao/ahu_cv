# coding: utf-8
# Project：ahu_cv
# Author：XHao
# Date ：2023-6-8 10:51
# Tool ：PyCharm

import numpy as np
import torch
from matplotlib import pyplot as plot
from matplotlib import animation


class XhaoPlot:
    def __init__(self):
        self.line = None

    def plot(self, X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
             ylim=None, xscale='linear', yscale='linear', fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5),
             axes=None):
        """绘制数据点"""
        if legend is None:
            legend = []
        # fig = plot.figure()
        axes = axes if axes else plot.subplot(111)
        X, Y = self.alignDimension(X, Y)
        axes.cla()    # 清除axes内容，
        # plt.cla()  # 清除axes，即当前 figure 中的活动的axes，但其他axes保持不变。
        # plt.clf()  # 清除当前 figure 的所有axes，但是不关闭这个 window，所以能继续复用于其他的 plot。
        # plt.close()  # 关闭figure window，如果没有指定figure，则指当前figure window。
        for x, y, fmt in zip(X, Y, fmts):
            if len(x):
                axes.plot(x, y, fmt)
            else:
                axes.plot(y, fmt)
        self.set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        return axes

    def alignDimension(self, X, Y):
        if self.has_one_axis(X):
            X = [X]
        if Y is None:
            X, Y = [[]] * len(X), X
        elif self.has_one_axis(Y):
            Y = [Y]
        if len(X) != len(Y):
            X = X * len(Y)
        return X, Y

    def set_axes(self, axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
        """设置matplotlib的轴"""

        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.set_xscale(xscale)
        axes.set_yscale(yscale)
        axes.set_xlim(xlim)
        axes.set_ylim(ylim)
        if legend:
            axes.legend(legend)
        axes.grid()

    def show(self):
        plot.show()

    def has_one_axis(self, X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list) and not hasattr(X[0], "__len__"))


if __name__ == '__main__':
    plt = XhaoPlot()
    x = np.arange(0, 3, 0.1)
    def f(x):
        return 3 * x ** 2 - 4 * x
    ax = plt.plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
    plt.show()
