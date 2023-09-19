# coding: utf-8
# Project：ahu_cv
# Author：XHao
# Date ：2023-6-30 16:00
# Tool ：PyCharm

import time
import numpy as np

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