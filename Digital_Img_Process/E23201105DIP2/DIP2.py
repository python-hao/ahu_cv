# coding=utf-8
'''
@Time     : 2023/12/23 09:36:03
@Author   : XHao
@Email    : 2510383889@qq.com
'''
# here put the import lib

import matplotlib.pyplot as plt
import cv2
import numpy as np
from copy import deepcopy


def scaleImgSize(origin_img, long_side=None, scale=None):
    h, w, _ = origin_img.shape
    if long_side:
        assert isinstance(long_side, int)
        # 长边缩放为long_side
        scale = float(long_side) / max(h, w)
    elif scale:
        assert isinstance(scale, int | float)
    else:
        raise ValueError("long_side or scale is needed at least one")
    new_w, new_h = int(w * scale), int(h * scale)
    resize_img = cv2.resize(origin_img, (new_w, new_h))
    return resize_img


class MedianFilter:
    def __init__(self, filename) -> None:
        self.img = scaleImgSize(cv2.imread(filename), 512)

    def addSaltNoise(self, snr):
        # 指定信噪比
        SNR = snr
        # 获取总共像素个数
        size = self.img.size
        img = deepcopy(self.img)
        # 因为信噪比是 SNR ，所以噪声占据百分之10，所以需要对这百分之10加噪声
        noiseSize = int(size * (1 - SNR))
        # 对这些点加噪声
        for k in range(0, noiseSize):
            # 随机获取 某个点
            xi = int(np.random.uniform(0, self.img.shape[1]))
            xj = int(np.random.uniform(0, self.img.shape[0]))
            # 增加噪声
            if img.ndim == 2:
                img[xj, xi] = 255
            elif img.ndim == 3:
                img[xj, xi] = 0
        return img

    def median_filter(self, kernel_size=3):
        """
        手动实现中值滤波器。

        :param kernel_size: 滤波器的大小，应为奇数
        :return: 滤波后的图像
        """
        # 边缘填充
        pad_size = kernel_size // 2
        padded_img = np.pad(self.img, ((pad_size, pad_size), (pad_size,
                            pad_size), (0, 0)), mode='constant', constant_values=0)

        # 输出的图像
        filtered_img = np.zeros_like(self.img)
        # 遍历图像的每个像素,c是通道，i是高，j是宽（h，w，c）
        for c in range(self.img.shape[2]):
            for i in range(self.img.shape[0]):
                for j in range(self.img.shape[1]):
                    # 获取当前像素的邻域
                    neighborhood = padded_img[i:i +
                                              kernel_size, j:j + kernel_size, c]
                    # 计算中值并赋值给过滤图像
                    filtered_img[i, j, c] = np.median(neighborhood)

        return filtered_img

    def adaptive_median_filter(self, kernel_size=5, max_kernel_size=11):
        """
        自适应核大小中值滤波器。

        :param kernel_size: 滤波器的初始（最小）核大小
        :param max_kernel_size: 滤波器的最大核大小
        :return: 滤波后的图像
        """

        # 输出的图像
        filtered_img = np.zeros_like(self.img)
        # 遍历图像的每个像素,c是通道，i是高，j是宽（h，w，c）
        for c in range(self.img.shape[2]):
            for i in range(self.img.shape[0]):
                for j in range(self.img.shape[1]):
                    kernel_size = 3  # 初始核大小
                    while kernel_size <= max_kernel_size:
                        # 计算当前核的半径
                        radius = kernel_size // 2

                        # 提取当前核的像素值
                        neighborhood = self.img[max(0, i - radius):min(self.img.shape[0], i + radius + 1),
                                                max(0, j - radius):min(self.img.shape[1], j + radius + 1), c]

                        # 计算核内的标准差
                        std_dev = np.std(neighborhood)

                        # 计算核内的中值
                        median = np.median(neighborhood)

                        # 判断当前像素是否为噪声点
                        if abs(self.img[i, j, c] - median) <= std_dev:
                            filtered_img[i, j, c] = median
                            break  # 如果不是噪声点，停止增加核的大小
                        else:
                            kernel_size += 2  # 增加核的大小（只考虑奇数大小的核）
        return filtered_img


class FFT:
    """
    自定义快速傅里叶变换 (FFT) 程序。
    """

    def __init__(self, input_signal=None) -> None:
        self.signal = input_signal

    def custom_fft(self, input_signal):
        """
        :param input_signal: 输入的一维信号
        :return: FFT 结果
        """
        n = len(input_signal)
        if n <= 1:
            return input_signal
        else:
            even = self.custom_fft(input_signal[::2])
            odd = self.custom_fft(input_signal[1::2])
            T = [np.exp(-2j * np.pi * k / n) * odd[k] for k in range(n // 2)]
            return [even[k] + T[k] for k in range(n // 2)] + [even[k] - T[k] for k in range(n // 2)]


def _1():
    """
    中值滤波
    """
    filename = r"E:\ahu_cv\Digital_Img_Process\E23201105-DIP1\assets\xhao.png"
    median_filter = MedianFilter(filename)
    # 加椒盐噪声
    img_salt = median_filter.addSaltNoise(snr=0.99)
    # 进行中值滤波
    img_flilter = median_filter.adaptive_median_filter(kernel_size=3)
    # 图像展示
    cv2.imshow('Origin', median_filter.img)
    cv2.imshow('Salt Noisy', img_salt)
    cv2.imshow('Remove Salt Noisy', img_flilter)
    cv2.moveWindow('Remove Salt Noisy', 0, 0)

    # cv2的基本操作
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _2():
    """
    自定义的快速傅里叶变换（FFT）,用来分析一个由两个不同频率的正弦波组合成的一维信号。
    第一个图展示了原始信号，它是两个正弦波的叠加，分别具有不同的频率。
    第二个图展示了该信号的FFT结果。

    FFT结果显示了两个明显的峰值，这两个峰值对应于输入信号中的两个正弦波的频率。
    这个示例展示了FFT在信号处理中的典型应用：分析信号的频率成分。
    通过FFT，我们可以识别出信号中包含的不同频率的波形。
    """
    # 创建一个测试信号 (例如一个简单的正弦波)
    t = np.linspace(0, 1, 256, endpoint=False)
    test_signal = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 10 * t)
    print(test_signal)
    # 应用自定义FFT
    fft_result = FFT().custom_fft(test_signal)

    # 绘图展示
    plt.figure(figsize=(12, 6))

    # 绘制原始信号
    plt.subplot(2, 1, 1)
    plt.plot(t, test_signal)
    plt.title("Original Signal")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

    # 绘制FFT结果
    plt.subplot(2, 1, 2)
    plt.plot(np.abs(fft_result))
    plt.title("FFT of the Signal")
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 以下两个二选一注释
    # 中值滤波
    # _1()
    # 快速傅里叶变换
    _2()
