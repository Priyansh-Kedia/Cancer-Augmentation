from segment_ycc import apply_mask
import cv2 as cv
import numpy as np

print("OpenCV version is {}".format(cv.__version__))

image = cv.imread("images/0+0/03cbe5d39233a65cd7f64a81094ccc54_p_0.tiff", -1)

NUCLEUS_RED = 100
NUCLEUS_GREEN = 100
NUCLEUS_BLUE = 130

# NUCLEUS_THRESHOLD = [128,0,128]
NUCLEUS_THRESHOLD = [90,255,cv.THRESH_BINARY]

NUCLEUS_LOW_RANGE = np.array([NUCLEUS_RED - 70, NUCLEUS_GREEN - 80, NUCLEUS_BLUE - 70])
NUCLEUS_HIGH_RANGE = np.array([NUCLEUS_RED + 70, NUCLEUS_GREEN + 80, NUCLEUS_BLUE + 60])

imageYCC = cv.cvtColor(image, cv.COLOR_BGR2YCR_CB)

result = apply_mask(imageYCC, image)
# result = cv.cvtColor(result, cv.COLOR_YCR_CB2BGR)
result = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
# cv.imshow("result", result)
# cv.waitKey()
# cv.destroyAllWindows()
ret, thresh = cv.threshold(result, NUCLEUS_THRESHOLD[0], NUCLEUS_THRESHOLD[1], NUCLEUS_THRESHOLD[2])
im2, contours = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# print("counts",len(contours))
# print(contours)
for c in contours:
    print(c)

result = cv.drawContours(result, contours, 0, (0,255,0), -1)

small, large = np.min(image), np.max(image)
print(small, large)

cv.imshow("result", thresh)
cv.waitKey()
cv.destroyAllWindows()
