import cv2 as cv
import numpy as np

image = cv.imread("images/0+0/03cbe5d39233a65cd7f64a81094ccc54_p_0.tiff", -1)

imageYCC = cv.cvtColor(image, cv.COLOR_BGR2YCR_CB)


RANGE = 25

LUMEN_RED = 210 #105
LUMEN_GREEN = 110 #105
LUMEN_BLUE = 110 #205


LUMEN_LOW_RANGE = np.array([LUMEN_RED - RANGE, LUMEN_GREEN - RANGE, LUMEN_BLUE - RANGE], np.uint8)
LUMEN_HIGH_RANGE = np.array([LUMEN_RED + RANGE, LUMEN_GREEN + RANGE, LUMEN_BLUE + RANGE], np.uint8)

# LUMEN_LOW_RANGE = np.array([0,133,77],np.uint8)
# LUMEN_HIGH_RANGE = np.array([235,173,127],np.uint8)


LUMEN_MASK = cv.inRange(imageYCC, LUMEN_LOW_RANGE, LUMEN_HIGH_RANGE)

result = cv.bitwise_and(imageYCC, imageYCC, mask=LUMEN_MASK)

result = imageYCC.copy()
result[LUMEN_MASK == 255] = [0,0,0]

cv.imshow("result",np.hstack([imageYCC,image, result]))
cv.waitKey()
cv.destroyAllWindows()