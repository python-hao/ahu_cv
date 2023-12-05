# coding: utf-8
# Project：ahu_cv
# Author：XHao
# Date ：2023-7-3 11:06
# Tool ：PyCharm
import time

from matplotlib import pyplot as plt
from matplotlib import animation


class XhaoPlot:
    def __init__(self):
        self.fig = None
        self.axes = []
        self.axes_setting = None

    def plot(self, X, Y, xlabel=None, ylabel=None, legend=None, xlim=None,
             ylim=None, xscale='linear', yscale='linear', fmts=('-', 'm--', 'g-.', 'r:'),
             rows=1, cols=1, figsize=(3.5, 2.5), axes=None):
        """绘制数据点"""
        if legend is None:
            legend = []
        if axes is None:
            _, axes = plt.subplots(rows, cols)
            self.axes.append(axes)
        self.fig = plt.figure(1, figsize=figsize)

        X, Y = self.alignDimension(X, Y)
        axes.cla()  # 清除axes内容，
        for x, y, fmt in zip(X, Y, fmts):
            length = min(len(x), len(y))
            line, = axes.plot(x[:length], y[:length], fmt)
        self.set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        # self.axes_setting = (axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        return axes

    def scatter(self, X, Y, s=None, c=None, alpha=0.5, xlabel=None, ylabel=None, legend=None, xlim=None,
                ylim=None, xscale='linear', yscale='linear', figsize=(3.5, 2.5),
                axes=None):
        if s is None:
            s = [20, ]
        if legend is None:
            legend = []
        axes = axes if axes else plt.subplot(111)
        X, Y = self.alignDimension(X, Y)
        # axes.cla()  # 清除axes内容，
        for x, y in zip(X, Y, ):
            if len(x):
                line = axes.scatter(x, y, s=s, alpha=alpha)
            else:
                line = axes.scatter(y, s=s, alpha=alpha)
        self.set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        return axes

    def subplot(self, *args):
        axes = plt.subplot(*args)
        self.axes.append(axes if hasattr(axes, '__len__') else axes)
        return axes

    def update(self, axes, X, Y):
        if axes is None:
            axes = self.axes
        X, Y = self.alignDimension(X, Y)
        l_lines = len(axes.lines)
        l_Y = len(Y)
        if l_Y == 0:
            return
        if l_lines < l_Y:
            for i in range(l_Y - l_lines):
                axes.plot([], [])
        lines = axes.lines
        # for i in range(min(l_lines, l_Y)):
        for i in range(l_Y):
            length = min(len(X[i]), len(X[i]))
            lines[i].set_data(X[i][:length], Y[i][:length])
            # axes.set_xlim(min(X[i]), max(X[i]))
            # axes.set_ylim(min(Y[i]), max(Y[i]))

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
        plt.show()

    def show_images(self, imgs, num_rows, num_cols, titles=None, scale=1.5):
        """绘制图像列表"""
        figsize = (num_cols * scale, num_rows * scale)
        figure, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
        axes = axes.flatten()
        for i, (ax, img) in enumerate(zip(axes, imgs)):
            if type(img).__name__ == 'Tensor':
                # 图⽚张量
                ax.imshow(img.numpy())
            else:
                # PIL图⽚
                ax.imshow(img)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            if titles:
                ax.set_title(titles[i])
        plt.show()
        return axes

    def has_one_axis(self, data):
        flag1 = hasattr(data, "ndim") and data.ndim == 1
        flag2 = False
        if isinstance(data, list):
            flag2 = True if len(data) == 0 else not hasattr(data[0], "__len__")
        # print(f'       flag1:{flag1}, flag2:{flag2}===={flag1 or flag2}')
        return flag1 or flag2


class AnimatorPlot:
    """在动画中绘制数据"""

    def __init__(self, fig=None, axes=None, X=None, Y=None, frames=60, repeat=True, interval=200, blit=False):
        self.axes_setting = None
        self.fig = fig if fig is not None else plt.figure()
        self.axes = axes if axes is not None else plt.subplot(111)
        self.axes.cla()
        self.axes.set_autoscale_on(True)
        self.x, self.y = X if X is not None else [], Y if Y is not None else []
        self.animator = animation.FuncAnimation(fig=self.fig, func=self.__animate, frames=frames, repeat=repeat,
                                                init_func=self.__init_data, interval=interval, blit=blit)

    def set_axes(self, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, xscale='linear', yscale='linear'):
        """设置matplotlib的轴"""
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.set_xscale(xscale)
        self.axes.set_yscale(yscale)
        self.axes.set_xlim(xlim)
        self.axes.set_ylim(ylim)
        if legend:
            self.axes.legend(legend)
        self.axes.grid()
        # self.axes_setting = (xlabel, ylabel, legend, xlim, ylim, xscale, yscale)

    def update(self, X, Y):
        X, Y = XhaoPlot().alignDimension(X, Y)
        l_lines = len(self.axes.lines)
        l_Y = len(Y)
        if l_lines < l_Y:
            for i in range(l_Y - l_lines):
                self.axes.plot([], [])
                # self.set_axes(*self.axes_setting)
            self.axes.set_autoscale_on(True)
        self.x, self.y = X, Y

    def __animate(self, x_list):
        lines = self.axes.lines
        if len(self.y) == 0:
            return lines,
        for i in range(len(self.y)):
            length = min(len(self.x[i]), len(self.y[i]))
            lines[i].set_data(self.x[i][:length], self.y[i][:length])
            # print(f'x:{self.x[i][:length]}, y:{self.y[i][:length]}')
        return lines,

    def __init_data(self):
        return self.axes.lines,

    def show(self):
        plt.show()


if __name__ == '__main__':
    import numpy as np
    import threading

    x = np.linspace(0, 2 * np.pi, 128)


    def job1(ani):
        y_ = []
        for index, i in enumerate(x):
            y_.append(np.sin(i))
            # print(f'\r{i}', end="")
            ani.update(x, [y_, np.array(y_) + 0.5, np.array(y_) + 1])
            # if index == 0:
            #     ani.axes.legend(['sin(X)', 'sin(X) + 1'])
            time.sleep(0.5)


    ani = AnimatorPlot(frames=60, interval=1)
    # ani.axes.set_xlim((min(x), max(x)))
    # ani.axes.set_ylim((-2, 2))
    th = threading.Thread(target=job1, args=(ani,))
    th.start()
    ani.show()
    # animator = XhaoPlot()
    # animator.plot([], [], xlabel='x', ylabel='p(x)', legend=['train_loss', 'train_acc', 'test_acc'])
    # for i in dir(animator.axes):
    #     print(i)
    # animator.show()
