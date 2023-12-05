# coding=utf-8
'''
@Time     : 2023/12/03 19:38:00
@Author   : XHao
@Email    : 2510383889@qq.com
'''
# here put the import lib

import cv2
import numpy as np

"""
cv2.imread()第二个参数说明:
   cv2.IMREAD_UNCHANGED = -1, //返回原通道原深度图像
   cv2.IMREAD_GRAYSCALE = 0, //返回单通道（灰度），8位图像
   cv2.IMREAD_COLOR = 1, //返回三通道，8位图像，为默认参数
   cv2.IMREAD_ANYDEPTH = 2, //返回单通道图像。如果原图像深度为16/32 位，则返回原深度，否则转换为8位
   cv2.IMREAD_ANYCOLOR = 4, //返回原通道，8位图像。

"""

# 定义一个装饰器，省的每次都写释放cv句柄的函数


def _releaseCV(waitKey=0):
    """
    waitKey: 0表示按下任意键就结束cv2.waitKey函数
    """
    def param_waraper(func):
        def wraper(*args, **kwargs):
            # print("start!!---")
            result = func(*args, **kwargs)
            cv2.waitKey(waitKey)
            cv2.destroyAllWindows()
            # print("end!!---")
            return result
        return wraper
    return param_waraper


@_releaseCV(waitKey=0)
def embed(target_img_path, add_imgs_path, result_path):
    # 检查嵌入图片的地址为列表形式
    if not isinstance(add_imgs_path, list):
        raise TypeError("add_imgs must be list like [path1, ]")

    # 按单通道（灰度），8位图像（0~255）读入
    # 之后缩放图片尺寸为（w，h），默认以线性插值方式
    # 把 add_img 归一化
    self_img_data = cv2.resize(cv2.imread(target_img_path, 0), (512, 512))
    h, w = self_img_data.shape
    add_imgs_data = [cv2.resize(cv2.imread(_img, 0), (w, h))
                     for _img in add_imgs_path]

    # 把add_img图像转为二值图
    add_imgs_data = [cv2.threshold(_img_data, 127, 1, cv2.THRESH_BINARY)[1].astype(np.uint8)
                     for _img_data in add_imgs_data]

    # 高6位不变, 低2位置为0(11111100==252,与252按位与就可以实现了)等待后面被嵌入
    img_252 = np.ones((w, h), dtype=np.uint8) * 252
    imgH7 = cv2.bitwise_and(img_252, self_img_data)

    # 签名嵌入第0位面
    embedded_img = cv2.bitwise_or(imgH7, add_imgs_data[0])

    # 专业嵌入第1位面(专业的二值图要与原图第1位对齐，故专业要左移1位)
    for row in range(h):
        for col in range(w):
            if add_imgs_data[1][row, col] == 1:
                add_imgs_data[1][row, col] = add_imgs_data[1][row, col] << 1
    embedded_img = cv2.bitwise_or(embedded_img, add_imgs_data[1])

    # 保存图像
    cv2.imwrite(result_path, embedded_img)

    # 显示图像
    cv2.imshow("self_img_data", self_img_data)
    cv2.imshow("embedded_img", embedded_img)


@_releaseCV(waitKey=0)
def disEmbed(embedded_img_path, bit_pos: [int] = [-1], save_dir=None):
    """
    bit_pos: 解码的位面，-1表示全解码
    """
    img_result = cv2.imread(embedded_img_path)
    h, w, c = img_result.shape
    # x用于将二进制数转换为十进制
    x = np.zeros((h, w, c, 8), dtype=np.uint8)
    for i in range(8):
        x[:, :, :, i] = 2**i
    r = np.zeros((h, w, c, 8), dtype=np.uint8)
    cv2.imshow("wait to be disembedded", img_result)
    bit_pos = range(8) if bit_pos == -1 else bit_pos
    for i in bit_pos:
        assert i >= 0
        r[:, :, :, i] = cv2.bitwise_and(img_result, x[:, :, :, i])
        mask = r[:, :, :, i] > 0
        r[mask] = 255
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(os.path.join(
                save_dir, f"{i}.png"), r[:, :, :, i])
        cv2.imshow(f"{i}", r[:, :, :, i])


if __name__ == "__main__":
    from pathlib import Path

    root_customer = Path(__file__).resolve().parent
    self_img_path = Path(root_customer / 'xhao.png').as_posix()
    add_img_path = [
        Path(root_customer / 'signature.png').as_posix(),
        Path(root_customer / 'subject.png').as_posix(),
    ]
    result_path = Path(root_customer/'DIP1_embed.png').as_posix()
    # 编码
    embed(self_img_path, add_img_path, result_path)

    # 解 码
    save_dir = Path(root_customer/'DIP1_disEmbed').as_posix()
    disEmbed(result_path, bit_pos=[0, 1], save_dir=save_dir)
