import tools
import numpy as np
import os
import cv2


def initName(innerclsnum, class_name):
    pic_kinds = [[] for i in range(len(class_name))]
    for i in range(innerclsnum):
        for pic_kind_idx in range(len(pic_kinds)):
            pic_kinds[pic_kind_idx].append(os.path.join(img_path, class_name[pic_kind_idx] + str(i + 1) + ".jpg"))
    return tuple(pic_kinds)


def getPictures(pic_paths: list):
    pic = []
    for path in pic_paths:
        pic.append(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
    return pic


def getCorners_info(imgs: list, show=False):
    # img = cv2.imread(img_path)
    # corner, rows, cols, img1 = tools.harris_detection(img, 512, 512, 0.01)
    # tools.showHarris(rows, cols, corner, img1)
    corners = []
    for img in imgs:
        corner, rows, cols, img1 = tools.harris_detection(img, 1024, 1024, 0.01)
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


def compareInnerClass(corners: list, imgs: list, grade_idx: list):
    if len(grade_idx) != 2:
        print("Wrong length of grade_idx(must be 2)!")
        return

    idx1, idx2 = grade_idx
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


def betterCompare(corners: list, imgs: list, idx: list):
    if len(idx) != 2:
        print("Wrong length of index(must be 2)!")
        return

    idx1, idx2 = idx
    features1, dir1 = tools.betterDescriptor(corners[idx1], imgs[idx1])
    features2, dir2 = tools.betterDescriptor(corners[idx2], imgs[idx2])

    y1, x1 = np.where(corners[idx1] > 0)
    y2, x2 = np.where(corners[idx2] > 0)

    matches, notConfidentMatches = tools.matchFeatures(features1, features2, 1)

    tools.plotImage(imgs[idx1], imgs[idx2], x1[matches[:, 1]], y1[matches[:, 1]], x2[matches[:, 2]], y2[matches[:, 2]])
    [precision, recall, F_score] = tools.accuracy(matches, notConfidentMatches, features1, features2)


if __name__ == "__main__":
    # from pathlib import Path
    # root_customer = Path(__file__).resolve().parent
    # img_path = root_customer / "Pictures"
    img_path = r"E:\ahu_cv\Digital_Img_Process\E23201081DIP3\Pictures"
    class_name = ["Me", "HardDisk", "card"]
    innerclsnum = 5

    paths = initName(innerclsnum, class_name)
    imgs = [getPictures(path_kind) for path_kind in paths]
    info = getCorners_info(imgs[2], False)
    corners = getCorners_only(info)

    compareInnerClass(corners, imgs[2], [1, 3])
    # betterCompare(corners, imgs[2], [1, 2])
