import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import CenterCrop
from scipy.ndimage import convolve


# Example usage
# img is the input image, dim1 and dim2 are the resized image dimensions, thresh is the threshold
# corners, rows, cols, img2 = harris_detection(img, dim1, dim2, thresh)
def harris_detection(img, dim1, dim2, thresh):
    img_torch = torch.tensor(img)
    crop = CenterCrop((dim1, dim2))
    image = crop(img_torch).numpy()

    gaus = cv2.getGaussianKernel(7, 1)
    deriv_gaus_x = cv2.getDerivKernels(1, 0, 7, normalize=True)
    deriv_gaus_y = cv2.getDerivKernels(0, 1, 7, normalize=True)

    ix = cv2.filter2D(image, -1, deriv_gaus_x[0] * deriv_gaus_x[1].T)
    iy = cv2.filter2D(image, -1, deriv_gaus_y[0] * deriv_gaus_y[1].T)
    ix2 = ix**2
    iy2 = iy**2
    ixiy = ix * iy

    ix2g = cv2.filter2D(ix2, -1, gaus)
    iy2g = cv2.filter2D(iy2, -1, gaus)
    ixiyg = cv2.filter2D(ixiy, -1, gaus)

    rows, cols = image.shape
    C = np.zeros((rows, cols))
    temp = 0

    for i in range(8, rows - 8):
        for j in range(8, cols - 8):
            M = np.array([[ix2g[i, j], ixiy[i, j]], [ixiyg[i, j], iy2g[i, j]]])
            C[i, j] = np.linalg.det(M) - 0.05 * (np.trace(M))**2
            if C[i, j] > temp:
                temp = C[i, j]

    corners = np.zeros((rows, cols))
    neighbours = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]])

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if C[i, j] > thresh * temp:
                for k in range(8):
                    if C[i, j] < C[i + neighbours[k, 0], j + neighbours[k, 1]]:
                        break
                if k == 7:
                    corners[i, j] = 1

    return corners, rows, cols, image

# Example usage:
# img = cv2.imread('your_image.jpg')
# corners, rows, cols, img2 = harris_detection(img, dim1, dim2, thresh)
# features, dir = descriptor(corners, img2)


def descriptor(corners, img2):
    # image = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gaus = cv2.getGaussianKernel(7, 1)
    deriv_gaus_x = np.gradient(gaus, axis=0, edge_order=0)
    deriv_gaus_y = deriv_gaus_x

    ix = convolve(img2, deriv_gaus_x)
    iy = convolve(img2, deriv_gaus_y)

    ixg = convolve(ix, gaus).astype(np.float32)
    iyg = convolve(iy, gaus).astype(np.float32)

    mag, dir = cv2.cartToPolar(ixg, iyg, angleInDegrees=True)

    dir[dir < 0] += 360

    for i in range(1, 9):
        dir[(dir >= (i - 1) * 45) & (dir <= i * 45)] = i  # direction quantize

    cornerY, cornerX = np.where(corners > 0)

    features = []

    for cornerIndex in range(len(cornerX)):
        startX = cornerX[cornerIndex] - 7
        startY = cornerY[cornerIndex] - 7
        endX = cornerX[cornerIndex] + 8
        endY = cornerY[cornerIndex] + 8

        window = dir[startY:endY, startX:endX]

        # squareMatrix = np.reshape(window, (4, 4, 4, 4))

        matAsVector = np.reshape(window, (1, -1))

        featureOfSquare = []

        for matIndex in range(len(matAsVector)):
            hist = np.zeros(8)
            temp = np.reshape(matAsVector[matIndex], (1, -1))

            for squareIndex in range(len(temp)):
                pixelDir = int(temp[0, squareIndex])
                hist[pixelDir] += 1

            featureOfSquare.extend(hist)

        featureOfSquare = np.divide(featureOfSquare, np.linalg.norm(featureOfSquare, ord=1))
        featureOfSquare[featureOfSquare >= 0.2] = 0.2
        featureOfSquare = np.divide(featureOfSquare, np.linalg.norm(featureOfSquare, ord=1))

        features.append(featureOfSquare)

    return features, dir

# Example usage:
# img = cv2.imread('your_image.jpg')
# corners, rows, cols, img2 = harris_detection(img, dim1, dim2, thresh)
# features, dirRatio = betterDescriptor(corners, img2)


def betterDescriptor(corners, img2):
    gaus = cv2.getGaussianKernel(7, 1)
    deriv_gaus_x = np.gradient(gaus, axis=0, edge_order=0)
    deriv_gaus_y = deriv_gaus_x

    ix = convolve(img2, deriv_gaus_x)
    iy = convolve(img2, deriv_gaus_y)

    ixg = convolve(ix, gaus).astype(np.float32)
    iyg = convolve(iy, gaus).astype(np.float32)

    mag, dir = cv2.cartToPolar(ixg, iyg, angleInDegrees=True)

    dir[dir < 0] += 360

    dirRatio = np.zeros_like(dir)

    for i in range(dir.shape[0]):
        for j in range(dir.shape[1]):
            for k in range(1, 9):
                if dir[i, j] >= (k - 1) * 45 and dir[i, j] <= k * 45:
                    dirRatio[i, j] = (dir[i, j] - (k - 1) * 45) / 45

    for i in range(1, 9):
        dir[dir >= (i - 1) * 45 & dir <= i * 45] = i  # direction quantize

    cornerY, cornerX = np.where(corners > 0)

    features = []

    for cornerIndex in range(len(cornerX)):
        startX = cornerX[cornerIndex] - 7
        startY = cornerY[cornerIndex] - 7
        endX = cornerX[cornerIndex] + 8
        endY = cornerY[cornerIndex] + 8

        window = dir[startY:endY, startX:endX]
        ratioWindow = dirRatio[startY:endY, startX:endX]
        magWindow = mag[startY:endY, startX:endX]

        # squareMatrix = np.reshape(window, (4, 4, 4, 4))
        # ratioMatrix = np.reshape(ratioWindow, (4, 4, 4, 4))
        # magMatrix = np.reshape(magWindow, (4, 4, 4, 4))

        matAsVector = np.reshape(window, (1, -1))
        ratioAsVector = np.reshape(ratioWindow, (1, -1))
        magAsVector = np.reshape(magWindow, (1, -1))

        featureOfSquare = []

        for matIndex in range(len(matAsVector)):
            hist = np.zeros(8)
            temp = np.reshape(matAsVector[matIndex], (1, -1))
            ratioTemp = np.reshape(ratioAsVector[matIndex], (1, -1))
            magTemp = np.reshape(magAsVector[matIndex], (1, -1))

            for squareIndex in range(len(temp)):
                pixelDir = int(temp[0, squareIndex])
                magInUpperBin = ratioTemp[0, squareIndex] * magTemp[0, squareIndex]
                magInLowerBin = (1 - ratioTemp[0, squareIndex]) * magTemp[0, squareIndex]

                hist[pixelDir] += magInUpperBin

                if pixelDir - 1 > 0:
                    hist[pixelDir - 1] += magInLowerBin

            featureOfSquare.extend(hist)

        featureOfSquare = np.divide(featureOfSquare, np.linalg.norm(featureOfSquare, ord=1))
        featureOfSquare[featureOfSquare >= 0.2] = 0.2
        featureOfSquare = np.divide(featureOfSquare, np.linalg.norm(featureOfSquare, ord=1))

        features.append(featureOfSquare)

    return features, dirRatio

# Example usage:
# features1 = np.array(...)  # Replace with your actual features1 data
# features2 = np.array(...)  # Replace with your actual features2 data
# thresh = 0.96
# matches, notConfidentMatches = matchFeatures(features1, features2, thresh)


def matchFeatures(features1, features2, thresh):
    features1 = np.stack(features1)
    features2 = np.stack(features2)
    matches = []
    notConfidentMatches = []

    for f1_idx in range(features1.shape[0]):
        dist = np.sqrt(np.sum((features1[f1_idx, :] - features2) ** 2, axis=1))
        sortedDistances = np.sort(dist)
        notConfidentMatches.append(sortedDistances[0])

        # # 检查次小距离是否为零，如果是，则认为匹配不置信
        # if sortedDistances[1] == 0:
        #     continue

        # 如果最小距离与次小距离的比例小于阈值，则认为是置信匹配
        print(f"比值：{sortedDistances[0] / sortedDistances[1] <= thresh}")
        if sortedDistances[1] != 0 and sortedDistances[0] / sortedDistances[1] <= thresh:
            f2_idx = np.where(dist == sortedDistances[0])[0][0]
            matches.append([f1_idx, f2_idx])
    # if len(matches) == 0:
    #     matches = [[0, 0]]
    return np.array(matches), np.array(notConfidentMatches)

# Example usage:
# showHarris(rows2, cols2, corners2, img2)


def showHarris(rows, cols, corners, img):
    plt.imshow(img)
    # plt.hold(True)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if corners[i, j] == 1:
                plt.plot(j, i, 'x', color='red')

    # plt.hold(False)
    plt.legend()
    plt.show()

# Example usage:
# img = cv2.imread('your_image.jpg')
# corners, rows, cols, img2 = testHarris(img, dim1, dim2, thresh)


def testHarris(img, dim1, dim2, thresh):
    img2 = cv2.resize(img, (dim2, dim1))

    image = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gaus = cv2.getGaussianKernel(7, 1)
    deriv_gaus_x = np.gradient(gaus)
    deriv_gaus_y = np.gradient(gaus)

    ix = cv2.filter2D(image, -1, deriv_gaus_x)
    iy = cv2.filter2D(image, -1, deriv_gaus_y)
    ix2 = ix * ix
    iy2 = iy * iy
    ixy = ix * iy

    ix2g = cv2.filter2D(ix2, -1, gaus)
    iy2g = cv2.filter2D(iy2, -1, gaus)
    ixiyg = cv2.filter2D(ixy, -1, gaus)

    rows, cols = image.shape
    C = np.zeros((rows, cols))
    temp = 0

    for i in range(8, rows - 8):
        for j in range(8, cols - 8):
            M = np.array([[ix2g[i, j], ixiyg[i, j]],
                          [ixiyg[i, j], iy2g[i, j]]])
            C[i, j] = np.linalg.det(M) - 0.05 * (np.trace(M)) ** 2
            if C[i, j] > temp:
                temp = C[i, j]

    corners = np.zeros((rows, cols))

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if C[i, j] > thresh * temp:
                startX = i - 1
                startY = j - 1
                endX = i + 1
                endY = j + 1
                window = C[startX:endX, startY:endY].flatten()

                for indexx in range(len(window)):
                    if C[i, j] < window[indexx]:
                        break

                if indexx == len(window) - 1:
                    corners[i, j] = 1

    return corners, rows, cols, img2

# Example usage:
# imgA_path = 'path_to_image_A.jpg'
# imgB_path = 'path_to_image_B.jpg'
# X1 = np.array([x1_1, x1_2, ...])  # Replace with actual X1 values
# Y1 = np.array([y1_1, y1_2, ...])  # Replace with actual Y1 values
# X2 = np.array([x2_1, x2_2, ...])  # Replace with actual X2 values
# Y2 = np.array([y2_1, y2_2, ...])  # Replace with actual Y2 values
# plotImage(imgA_path, imgB_path, X1, Y1, X2, Y2)


def plotImage(imgA, imgB, X1=None, Y1=None, X2=None, Y2=None, only_stack=False):
    imgA = cv2.normalize(imgA, None, 0, 1, cv2.NORM_MINMAX)
    imgB = cv2.normalize(imgB, None, 0, 1, cv2.NORM_MINMAX)

    height, width = imgA.shape[0], imgA.shape[1] + imgB.shape[1]

    newImage = np.hstack((imgA, imgB))
    plt.imshow(newImage)
    if not only_stack:
        for i in range(X1.shape[0]):
            color = np.random.rand(3, 1)
            plt.plot([X1[i], imgA.shape[1] + X2[i]], [Y1[i], Y2[i]], '*-', color=color, linewidth=2)

    plt.show()

# Example usage:
# imgA_path = 'path_to_image_A.jpg'
# imgB_path = 'path_to_image_B.jpg'
# X1 = np.array([x1_1, x1_2, ...])  # Replace with actual X1 values
# Y1 = np.array([y1_1, y1_2, ...])  # Replace with actual Y1 values
# X2 = np.array([x2_1, x2_2, ...])  # Replace with actual X2 values
# Y2 = np.array([y2_1, y2_2, ...])  # Replace with actual Y2 values
# plotImagePart4(imgA_path, imgB_path, X1, Y1, X2, Y2)


def plotImagePart4(imgA, imgB, X1, Y1, X2, Y2):
    h = 0

    imgA = cv2.normalize(cv2.imread(imgA), None, 0, 1, cv2.NORM_MINMAX)
    imgB = cv2.normalize(cv2.imread(imgB), None, 0, 1, cv2.NORM_MINMAX)

    height, width = imgA.shape[0], imgA.shape[1] + imgB.shape[1]

    newImage = np.hstack((imgA, imgB))
    plt.imshow(newImage)

    for i in range(1, X1.shape[0]):
        plt.plot([Y1[i - 1], Y1[i - 1]], [X1[i - 1], X1[i]], '*-', color='black', linewidth=2)
        plt.plot([Y2[i - 1] + imgA.shape[1], Y2[i - 1] + imgA.shape[1]], [X2[i - 1], X2[i]], '*-', color='black', linewidth=2)

    plt.show()

    return h

# Example usage:
# precision, recall, F_score = accuracy(matches, notConfidentMatches, features1, features2)


def accuracy(matches, notConfidentMatches, features1, features2):
    mr, mc, _ = matches.shape
    rr, rc, _ = notConfidentMatches.shape
    fr1, fc1, _ = features1.shape
    fr2, fc2, _ = features2.shape

    diff = max(fr1, fr2)

    # Calculate True Positive, False Negative, False Positive, True Negative
    TP = rr
    FN = abs(min(fr1, fr2) - rr)
    FP = abs(min(fr1, fr2) - rr)
    TN = abs(diff - rr)

    precision = TP / (FP + TP)
    recall = TP / (TP + FN)

    F_score = 2 / ((recall ** -1) + (precision ** -1))

    return precision, recall, F_score
