from numba import jit
from pre_learning.xhaoTools import d2l
import numpy as np
from timeit import timeit

@jit(nopython=True) # jit，numba装饰器中的一种
def go_numba(): # 首次调用时，函数被编译为机器代码
    x = np.arange(100)
    y = np.arange(100)
    z = x**2 + y**3
    return z

def go_numpy():
    x = np.arange(100)
    y = np.arange(100)
    z = x**2 + y**3
    return z

def go_pylist():
    a = list(range(100))
    b = list(range(100))
    c = []
    for i in range(len(a)):
        a[i] = i ** 2
        b[i] = i ** 3
        c.append(a[i] + b[i])
    return c

with d2l.Benchmark("list"):
    for i in range(1000):
        go_pylist()

with d2l.Benchmark("numpy"):
    for i in range(1000):
        go_numpy()

with d2l.Benchmark("numba + numpy 首次编译"):
    for i in range(1000):
        go_numba()

with d2l.Benchmark("numba + numpy 再次编译"):
    for i in range(1000):
        go_numba()
        
# print('python List:', timeit('go_pylist()', 'from __main__ import go_pylist', number=1000))
# print('python numpy:', timeit('go_numpy()', 'from __main__ import go_numpy', number=1000))
# print('python numba + numpy 首次编译:', timeit('go_numba()', 'from __main__ import go_numba', number=1000))
# print('python numba + numpy 再次运行:', timeit('go_numba()', 'from __main__ import go_numba', number=1000))
