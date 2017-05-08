import numpy as np
import cv2
import sys


def _thinningIteration(image, iter):
    marker = np.zeros(image.shape, dtype=np.uint8)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            p2 = image[i-1, j]
            p3 = image[i-1, j+1]
            p4 = image[i, j+1]
            p5 = image[i+1, j+1]
            p6 = image[i+1, j]
            p7 = image[i+1, j-1]
            p8 = image[i, j-1] 
            p9 = image[i-1, j-1]  

            C  = (not p2 and (p3 or p4)) + (not p4 and (p5 or p6)) + \
                     (not p6 and (p7 or p8)) + (not p8 and (p9 or p2))
            N1 = (p9 or p2) + (p3 or p4) + (p5 or p6) + (p7 or p8)
            N2 = (p2 or p3) + (p4 or p5) + (p6 or p7) + (p8 or p9)
            N  = min(N1, N2)
            m = 0
            if iter == 0:
                m = ((p6 or p7 or not p9) and p8)
            else:
                m = ((p2 or p3 or not p5) and p4)

            if C == 1 and (N >= 2 and N <= 3) and m == 0:
                marker[i,j] = 1;

    return image * (1 - marker)


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
	if src == None:
		sys.exit()
	bw = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
	_, bw2 = cv2.threshold(bw, 10, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	bw2 = thinning(255 - bw2)
	cv2.imshow("src", bw)
	cv2.imshow("thinning", bw2)
	cv2.waitKey()

