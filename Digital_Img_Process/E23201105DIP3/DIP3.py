# coding=utf-8
'''
@Time     : 2023/12/24 15:43:13
@Author   : XHao
@Email    : 2510383889@qq.com
'''
# here put the import lib

import tools
import numpy as np
import os
import cv2


def initName(cfg):
    pic_kinds = [[] for i in range(len(cfg["class_name"]))]
    for i in range(cfg["innerclsnum"]):
        for pic_kind_idx in range(len(pic_kinds)):
            pic_kinds[pic_kind_idx].append(os.path.join(cfg["img_path"], cfg["class_name"][pic_kind_idx] + str(i + 1) + ".jpg"))
    return tuple(pic_kinds)


def getPictures(pic_paths: list):
    pic = []
    for path in pic_paths:
        pic.append(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
    return pic


def getCorners_info(imgs: list, cfg: dict, show=False):
    """
    param imgs: 待进行Harris角点检测的若干输入图像
    param show: 是否显示harris角点图像
    return corners: 一个二值图像,表示检测到的角点
    """
    imgs = imgs[cfg["curIdx_class"]]  # 待进行Harris角点检测的若干输入图像
    corners = []
    for img in imgs:
        # rows 和 cols 表示输入图像的维度，img1 是中心裁剪的输入图像
        corner, rows, cols, img1 = tools.harris_detection(img, 1024, 1024, 0.01)
        print(f"corner shape:{corner.shape}, center-cropped shape:{img1.shape}")
        corners.append([corner, rows, cols, img1])
    if show:
        for cor in corners:
            tools.showHarris(cor[1], cor[2], cor[0], cor[3])
    return corners


def getCorners_only(cor: list):
    corners = []
    for item in cor:
        corners.append(item[0])
    return corners


def compareInnerClass(corners: list, imgs: list, cfg: dict):
    grade_idx = cfg["compare_grade"]
    if len(grade_idx) != 2:
        print("Wrong length of grade_idx(must be 2)!")
        return
    idx1, idx2 = grade_idx
    imgs = imgs[cfg["curIdx_class"]]  # 待进行Harris角点特征匹配的原输入图像
    features1, dir1 = tools.descriptor(corners[idx1], imgs[idx1])
    features2, dir2 = tools.descriptor(corners[idx2], imgs[idx2])

    y1, x1 = np.where(corners[idx1] > 0)
    y2, x2 = np.where(corners[idx2] > 0)

    matches, notConfidentMatches = tools.matchFeatures(features1, features2, 1)
    # matches[:, 2]表示在features2中匹配的
    try:
        tools.plotImage(imgs[idx1], imgs[idx2], x1[matches[:, 1]], y1[matches[:, 1]], x2[matches[:, 2]], y2[matches[:, 2]])
        [precision, recall, F_score] = tools.accuracy(matches, notConfidentMatches, features1, features2)
    except IndexError:
        tools.plotImage(imgs[idx1], imgs[idx2], only_stack=True)
    finally:
        pass


def betterCompare(corners: list, imgs: list, cfg: dict):
    grade_idx = cfg["compare_grade"]
    if len(grade_idx) != 2:
        print("Wrong length of grade_idx(must be 2)!")
        return
    idx1, idx2 = grade_idx
    imgs = imgs[cfg["curIdx_class"]]  # 待进行Harris角点特征匹配的原输入图像
    features1, dir1 = tools.betterDescriptor(corners[idx1], imgs[idx1])
    features2, dir2 = tools.betterDescriptor(corners[idx2], imgs[idx2])

    y1, x1 = np.where(corners[idx1] > 0)
    y2, x2 = np.where(corners[idx2] > 0)

    matches, notConfidentMatches = tools.matchFeatures(features1, features2, cfg["compare_thresh"])

    tools.plotImage(imgs[idx1], imgs[idx2], x1[matches[:, 1]], y1[matches[:, 1]], x2[matches[:, 2]], y2[matches[:, 2]])
    [precision, recall, F_score] = tools.accuracy(matches, notConfidentMatches, features1, features2)


if __name__ == "__main__":
    # 1、定义参数
    config = {
        "img_path": r"F:\Programs\ahu_cv\Digital_Img_Process\E23201081DIP3\Pictures",
        "class_name": ["Me", "HardDisk", "card"],
        "innerclsnum": 5,
        "curIdx_class": 2,
        "compare_grade": [1, 4],
        "compare_thresh": 1
    }

    # 2、初始化图片，读取图片为矩阵数组（一个图片对应一个矩阵）
    paths = initName(config)
    imgs = [getPictures(path_kind) for path_kind in paths]

    # 3、获取Harris角点
    info = getCorners_info(imgs, config, show=True)
    corners = getCorners_only(info)
    print()
    # 4、匹配算法：两张图进行特征匹配，画出匹配连线图
    compareInnerClass(corners, imgs, config)

    # 5、改进的匹配算法：两张图进行特征匹配，画出匹配连线图
    # betterCompare((corners, imgs, config)
