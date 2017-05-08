import numpy as np
import cv2
import sys


def _g(X, Y, Z):
    return np.logical_and(np.logical_not(X), np.logical_or(Y, Z))


def _f(A, B, C, D):
    return np.logical_and(np.logical_or(A, np.logical_or(B, np.logical_not(C))), D)


def _thinningIteration(image, iter):
    h, w = image.shape
    P2 = image[0:h-2, 1:w-1]
    P3 = image[0:h-2, 2:w]
    P4 = image[1:h-1, 2:w]
    P5 = image[2:h, 2:w]
    P6 = image[2:h, 1:w-1]
    P7 = image[2:h, 0:w-2]
    P8 = image[1:h-1, 0:w-2]
    P9 = image[0:h-2, 0:w-2]
    C = _g(P2, P3, P4).astype(np.uint8) + _g(P4, P5, P6).astype(np.uint8) + \
        _g(P6, P7, P8).astype(np.uint8) + _g(P8, P9, P2).astype(np.uint8)
    N1 = np.logical_or(P9, P2).astype(np.uint8) + np.logical_or(P3, P4).astype(np.uint8) + \
         np.logical_or(P5, P6).astype(np.uint8) + np.logical_or(P7, P8).astype(np.uint8)
    N2 = np.logical_or(P2, P3).astype(np.uint8) + np.logical_or(P4, P5).astype(np.uint8) + \
         np.logical_or(P6, P7).astype(np.uint8) + np.logical_or(P8, P9).astype(np.uint8)
    N = np.minimum(N1, N2)
    M = _f(P2, P3, P5, P4) if iter else _f(P6, P7, P9, P8)
    MR = np.zeros((h, w))
    MR[1:h-1, 1:w-1] = np.logical_and(
                           np.logical_and(
                               np.logical_and(
                                   np.where(C == 1, 1, 0), 
                                   np.where(N >= 2, 1, 0)), 
                               np.where(N <= 3, 1, 0)), 
                           np.where(M == 0, 1, 0))
    return image * (1 - MR)


def thinning(src):
    dst = src.copy() / 255
    prev = np.zeros(src.shape[:2], np.uint8)
    diff = None

    while True:
        dst = _thinningIteration(dst, 0)
        dst = _thinningIteration(dst, 1)
        diff = np.absolute(dst - prev)
        prev = dst.copy()
        if np.sum(diff) == 0:
            break
    return dst * 255


if __name__ == "__main__":
    src = cv2.imread("kanji.bmp")
    if src is None:
        sys.exit()
    bw = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _, bw2 = cv2.threshold(bw, 10, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw2 = thinning(255 - bw2)
    cv2.imshow("src", bw)
    cv2.imshow("thinning", bw2)
    cv2.waitKey()

