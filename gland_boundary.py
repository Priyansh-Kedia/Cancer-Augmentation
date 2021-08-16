from segment_ycc import apply_mask
import cv2 as cv
import numpy as np
import cv2

from matplotlib import pyplot as plt

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
g = cv.cvtColor(image,cv.COLOR_RGB2GRAY)
edge = cv.Canny(g, 60, 180)
fig, ax = plt.subplots(1, figsize=(12,8))
# out = image.copy()
# contours = cv.findContours(edge, 
#                             cv.RETR_EXTERNAL,
#                             cv.CHAIN_APPROX_NONE)
# print(len(contours))
# # cv.drawContours(image, contours[0], -1, (0,0,0), thickness = 2)
# contours, h = cv.findContours(edge, 
#                                cv.RETR_LIST,
#                                cv.CHAIN_APPROX_NONE)
# contours = sorted(contours, key=cv.contourArea, reverse=True)
# cv.drawContours(image, contours[0], -1, (0,0,255), thickness = 5)
# plt.imshow(edge, cmap='Greys')

# plt.show()
image = apply_mask(imageYCC, image)
# cv.imshow('once', result)

dimensions = image.shape

height= image.shape[0]
width = image.shape[1]
size = height*width

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)
cv.imshow("gray", gray)

_,thresh = cv2.threshold(gray,240,255,cv2.THRESH_BINARY)

cnts, hier = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
size_elements = 0
print("cnts", len(cnts))
cnts.pop()
cnts = [c for c in cnts if (cv.contourArea(c) > 500.0 and cv.contourArea(c) < 7000.0) ]
cv2.drawContours(image,cnts, -1, (0, 0, 255), thickness=cv.FILLED)

cv2.imshow("Image", image)
print("size elements total : ", size_elements)
print("size of pic : ", size)
print("rate of fullness : % ", (size_elements/size)*100)
cv2.waitKey(0)

# result = cv.cvtColor(result, cv.COLOR_YCR_CB2BGR)
result = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
# cv.imshow("result", result)
# cv.waitKey()
# cv.destroyAllWindows()
blurred = cv.GaussianBlur(result, (5, 5), 0)
ret, thresh = cv.threshold(result, NUCLEUS_THRESHOLD[0], NUCLEUS_THRESHOLD[1], NUCLEUS_THRESHOLD[2])
# im2, contours = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# print("counts",len(contours))
# print(contours)

# thresh[thresh == 255] = [0]
contours = sorted(contours, key=cv.contourArea, reverse=False)
# print(len(contours))
result = cv.drawContours(result, contours[0], -1, (0,255,0), -2)

small, large = np.min(thresh), np.max(thresh)
print(small, large)

# result = cv.cvtColor(result, cv.COLOR_GRAY2BGR)
# out = edge.copy()
ret, out = cv.threshold(edge, 240, 255, 1)
contours, hierarchy = cv.findContours(out, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)


contours = sorted(contours, key=cv.contourArea, reverse=False)
# print(len(contours))
results = cv.drawContours(edge, contours[0], -1, (0,0,0), -2)
mask = cv.inRange(edge, 240, 255)
print(np.min(edge), np.max(edge))
# edge[mask == 0] = 2132

kernel = np.ones((4,4), np.uint8)
# erosion = cv.erode(ed,kernel,iterations = 1)
img_dilation = cv.dilate(edge, kernel, iterations=1)
img_dilation = cv.dilate(img_dilation, kernel, iterations=1)
img_not = cv.bitwise_not(img_dilation)
print(img_not)
# ret, out = cv.threshold(img_not, 0, 10, 1)
ret, out = cv.threshold(img_not, 50, 255, cv.THRESH_BINARY)
contours, hierarchy = cv.findContours(out, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


contours = sorted(contours, key=cv.contourArea, reverse=True)
print(len(contours))
edge = img_not.copy()
# contours.pop()
contours = contours[0:2]
print(cv.contourArea(contours[0]), cv.contourArea(contours[1]))
print(len(contours))
edge = cv.drawContours(edge, contours, -1, (1,0,0), -2)
image[edge == 255] = [200, 123, 123]
# cv.imshow("result", np.hstack([img_dilation, img_not, edge]))
cv.imshow("result", np.hstack([image]))

cv.waitKey()
cv.destroyAllWindows()
