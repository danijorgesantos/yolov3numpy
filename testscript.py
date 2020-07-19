import cv2
import numpy as np
from os.path import isfile, join

# mypath = 'train'

# img = cv2.imread(join(mypath, 'Toyota-Yaris-01-696x464.jpg'),0)
# rows,cols = img.shape

# print(img.shape)

# array = []

# array.append(img)

# print(array)

# for i in range(rows):
#     for j in range(cols):
#         k = img[i,j]


# #show it
# cv2.imshow("some window", img)
# cv2.waitKey(0)

img = np.uint8(np.random.random((464, 696,3))*255)
dst = cv2.integral(img)

cv2.imshow("some window", img)
cv2.waitKey(0)