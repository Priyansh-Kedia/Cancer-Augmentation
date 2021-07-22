import cv2 as cv
import numpy as np

"""
Cytoplasm classification
"""
CYTOPLASM_RED = 195
CYTOPLASM_GREEN = 180
CYTOPLASM_BLUE = 210

# CYTOPLASM_RED = 150
# CYTOPLASM_GREEN = 150
# CYTOPLASM_BLUE = 180

image = cv.imread("images/0+0/03cbe5d39233a65cd7f64a81094ccc54_p_0.tiff", -1)
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
image_hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV)
RANGE = 25
CYTOPLASM_LOW_RANGE = np.array([CYTOPLASM_RED - RANGE, CYTOPLASM_GREEN - RANGE, CYTOPLASM_BLUE - RANGE])
CYTOPLASM_HIGH_RANGE = np.array([CYTOPLASM_RED + RANGE, CYTOPLASM_GREEN + RANGE, CYTOPLASM_BLUE + RANGE])
mask = cv.inRange(image, CYTOPLASM_LOW_RANGE, CYTOPLASM_HIGH_RANGE)
result = cv.bitwise_and(image, image, mask=mask)
cv.imwrite("images/{}".format("03cbe5d39233a65cd7f64a81094ccc54_p_0_marked.tiff"), image)
# cv.imshow("original", image)
result = image.copy()
result[mask == 255] = [0,255,0]
cv.imshow("result",np.hstack([image, result]))

# cv.imshow("result", image)
cv.waitKey()
cv.destroyAllWindows()