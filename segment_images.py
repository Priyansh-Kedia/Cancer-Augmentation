import cv2 as cv
import numpy as np

"""
Cytoplasm classification
"""
LUMEN_RED = 210
LUMEN_GREEN = 210
LUMEN_BLUE = 210

NUCLEUS_RED = 80
NUCLEUS_GREEN = 69
NUCLEUS_BLUE = 129

CYTOPLASM_RED = 145 # 195.180.210
CYTOPLASM_GREEN = 130
CYTOPLASM_BLUE = 160

image = cv.imread("images/0+0/03cbe5d39233a65cd7f64a81094ccc54_p_0.tiff", -1)
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
image_hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV)

RANGE = 25
CYTOPLASM_RANGE = 75

LUMEN_LOW_RANGE = np.array([LUMEN_RED - RANGE, LUMEN_GREEN - RANGE, LUMEN_BLUE - RANGE])
LUMEN_HIGH_RANGE = np.array([LUMEN_RED + RANGE, LUMEN_GREEN + RANGE, LUMEN_BLUE + RANGE])

NUCLEUS_LOW_RANGE = np.array([NUCLEUS_RED - RANGE, NUCLEUS_GREEN - RANGE, NUCLEUS_BLUE - RANGE])
NUCLEUS_HIGH_RANGE = np.array([NUCLEUS_RED + RANGE, NUCLEUS_GREEN + RANGE, NUCLEUS_BLUE + RANGE])

CYTOPLASM_LOW_RANGE = np.array([CYTOPLASM_RED - CYTOPLASM_RANGE, CYTOPLASM_GREEN - CYTOPLASM_RANGE, CYTOPLASM_BLUE - CYTOPLASM_RANGE])
CYTOPLASM_HIGH_RANGE = np.array([CYTOPLASM_RED + CYTOPLASM_RANGE, CYTOPLASM_GREEN + CYTOPLASM_RANGE, CYTOPLASM_BLUE + CYTOPLASM_RANGE])

LUMEN_MASK = cv.inRange(image, LUMEN_LOW_RANGE, LUMEN_HIGH_RANGE)
NUCLEUS_MASK = cv.inRange(image, NUCLEUS_LOW_RANGE, NUCLEUS_HIGH_RANGE)
CYTOPLASM_MASK = cv.inRange(image, CYTOPLASM_LOW_RANGE, CYTOPLASM_HIGH_RANGE)

LUMEN_MASK = LUMEN_MASK + NUCLEUS_MASK + CYTOPLASM_MASK

result = cv.bitwise_and(image, image, mask=LUMEN_MASK)
cv.imwrite("images/{}".format("03cbe5d39233a65cd7f64a81094ccc54_p_0_marked.tiff"), image)
# cv.imshow("original", image)
result = image.copy()
result[LUMEN_MASK == 255] = [0,255,0]
result[NUCLEUS_MASK == 255] = [255,0,0]
result[CYTOPLASM_MASK == 255] = [0, 0, 0]
cv.imshow("result",np.hstack([image, result]))

# cv.imshow("result", image)
cv.waitKey()
cv.destroyAllWindows()