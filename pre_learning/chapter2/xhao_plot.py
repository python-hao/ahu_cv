# coding: utf-8
# Project：ahu_cv
# Author：XHao
# Date ：2023-6-9 17:17
# Tool ：PyCharm
import math

import numpy as np
from matplotlib import pyplot as plot
from matplotlib import animation


# fig, ax = plt.subplots()
#
# x = np.arange(0, 2 * np.pi, 0.01)
# # 因为这里返回的是一个列表，但是我们只想要第一个值
# # 所以这里需要加,号
# line, = ax.plot(x, np.sin(x))
#
#
# def animate(i):
#     line.set_ydata(np.sin(x + i / 10.0))  # update the data
#     return line,
#
#
# def init():
#     line.set_ydata(np.sin(x))
#     return line,
#
#
# # call the animator.  blit=True means only re-draw the parts that have changed.
# # blit=True dose not work on Mac, set blit=False
# # interval= update frequency
# # frames帧数
# ani = animation.FuncAnimation(fig=fig, func=animate, frames=100, init_func=init,
#                               interval=1, blit=False)
#
# plt.show()


class XhaoPlot:
    def __init__(self):
        pass

    def plot(self, X, Y, xlabel=None, ylabel=None, legend=None, xlim=None,
             ylim=None, xscale='linear', yscale='linear', fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5),
             axes=None):
        """绘制数据点"""
        if legend is None:
            legend = []
        # fig = plot.figure()
        axes = axes if axes else plot.subplot(111)
        X, Y = self.alignDimension(X, Y)
        axes.cla()  # 清除axes内容，
        for x, y, fmt in zip(X, Y, fmts):
            if len(x):
                line, = axes.plot(x, y, fmt)
            else:
                line, = axes.plot(y, fmt)
        self.set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        return axes

    def scatter(self, X, Y, s=None, c=None, alpha=0.5, xlabel=None, ylabel=None, legend=None, xlim=None,
                ylim=None, xscale='linear', yscale='linear', figsize=(3.5, 2.5),
                axes=None):
        if s is None:
            s = [20, ]
        if legend is None:
            legend = []
        axes = axes if axes else plot.subplot(111)
        X, Y = self.alignDimension(X, Y)
        # axes.cla()  # 清除axes内容，
        for x, y in zip(X, Y,):
            if len(x):
                axes.scatter(x, y, s=s, alpha=alpha)
            else:
                axes.scatter(y, s=s, alpha=alpha)
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
    x = np.arange(-7, 7, 0.1)


    def f(x):
        return 3 * x ** 2 - 4 * x


    def normal(x, mu, sigma):
        p = 1 / math.sqrt(2 * math.pi * sigma ** 2)
        return p * np.exp(-0.5 / sigma ** 2 * (x - mu) ** 2)


    params = [(0, 1), (0, 2), (1, 1)]
    # axes = plt.plot(x, [normal(x, mu, sigma) for mu, sigma in params], 'x', 'p(x)',
    #                       legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])
    ax = plt.scatter(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x', ylabel='p(x)',
                     legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])
    plt.show()
