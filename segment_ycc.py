import cv2 as cv
import numpy as np



RANGE = 25
CYTOPLASM_RANGE = 10

LUMEN_RED = 210 #105
LUMEN_GREEN = 110 #105
LUMEN_BLUE = 110 #205

CYTOPLASM_RED = 150 # 195.180.210
CYTOPLASM_GREEN = 150  #  (108, 140, 173);
CYTOPLASM_BLUE = 150 # (144, 147, 163);

NUCLEUS_RED = 100
NUCLEUS_GREEN = 100
NUCLEUS_BLUE = 130

LUMEN_LOW_RANGE = np.array([LUMEN_RED - RANGE, LUMEN_GREEN - RANGE, LUMEN_BLUE - RANGE], np.uint8)
LUMEN_HIGH_RANGE = np.array([LUMEN_RED + RANGE, LUMEN_GREEN + RANGE, LUMEN_BLUE + RANGE], np.uint8)

CYTOPLASM_LOW_RANGE = np.array([CYTOPLASM_RED - 25, CYTOPLASM_GREEN - 50, CYTOPLASM_BLUE - CYTOPLASM_RANGE])
CYTOPLASM_HIGH_RANGE = np.array([CYTOPLASM_RED + 25, CYTOPLASM_GREEN + 50, CYTOPLASM_BLUE + CYTOPLASM_RANGE])

NUCLEUS_LOW_RANGE = np.array([NUCLEUS_RED - 70, NUCLEUS_GREEN - 80, NUCLEUS_BLUE - 70])
NUCLEUS_HIGH_RANGE = np.array([NUCLEUS_RED + 70, NUCLEUS_GREEN + 80, NUCLEUS_BLUE + 60])



def apply_mask(imageYCC, image):
    LUMEN_MASK = cv.inRange(imageYCC, LUMEN_LOW_RANGE, LUMEN_HIGH_RANGE)
    CYTOPLASM_MASK = cv.inRange(imageYCC, CYTOPLASM_LOW_RANGE, CYTOPLASM_HIGH_RANGE)
    NUCLEUS_MASK = cv.inRange(imageYCC, NUCLEUS_LOW_RANGE, NUCLEUS_HIGH_RANGE)

    LUMEN_MASK = LUMEN_MASK + NUCLEUS_MASK + CYTOPLASM_MASK

    result = cv.bitwise_and(imageYCC, imageYCC, mask=LUMEN_MASK)

    result = image.copy()
    result[LUMEN_MASK == 255] = [255,255,255]
    result[NUCLEUS_MASK == 255] = [128,0,128]
    result[CYTOPLASM_MASK == 255] = [255,105,180]

    # cv.imshow("result",np.hstack([imageYCC,image, result]))
    # cv.waitKey()
    # cv.destroyAllWindows()

    return result


if __name__ == "__main__":
    image = cv.imread("images/0+0/03cbe5d39233a65cd7f64a81094ccc54_p_0.tiff", -1)
    
    imageYCC = cv.cvtColor(image, cv.COLOR_BGR2YCR_CB)
    apply_mask(imageYCC, image)