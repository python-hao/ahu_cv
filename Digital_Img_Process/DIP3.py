# coding=utf-8
'''
@Time     : 2023/12/24 15:43:13
@Author   : XHao
@Email    : 2510383889@qq.com
'''
# here put the import lib


import matplotlib.pyplot as plt
import cv2
import numpy as np
from copy import deepcopy

import cv2
import numpy as np

a = np.arange(500, 100, -1, dtype=np.int64).reshape(20, -1)
b = np.arange(100, 500, 1, dtype=np.int64).reshape(20, -1)
dist = np.sqrt(np.sum((a[0, :] - b) ** 2, axis=1))
mat = np.sort(dist)
# search = np.where(dist == mat[0])[0][0]
if mat[0] / mat[1] < 0.88:
    print(mat[0], mat[1])
# print(a[search])
