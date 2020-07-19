import numpy as np
import cv2

#read image
img_src = cv2.imread('mer.png')

def initializeFilter(size, scale = 1.0):
    stddev = scale/np.sqrt(np.prod(size))
    return np.random.normal(loc = 0, scale = stddev, size = size)

f1 = initializeFilter(25).reshape(5,5)

#filter the source image
img_rst = cv2.filter2D(img_src,-1,f1)

#save result image
cv2.imwrite('result.jpg',img_rst)