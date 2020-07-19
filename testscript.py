import cv2
import numpy as np
from os.path import isfile, join

mypath = 'train'

img = cv2.imread(join(mypath, 'Toyota-Yaris-01-696x464.jpg'),0)
rows,cols = img.shape


n_c, in_dim = img.shape # image dimensions, n_c = 464, in_dim = 696

(n_f, n_c_f, f, _) = filt.shape # filter dimensions


array = []
array2= []

array.append(img)

#print(array)

for i in range(rows):
    for j in range(cols):
        k = img[i,j]
        array2.append(k)


# out = np.zeros((n_f,out_dim,out_dim))

# # convolve the filter over every part of the image, adding the bias at each step. 
# for curr_f in range(n_f):
#     curr_y = out_y = 0
#         while curr_y + f <= in_dim:
#             curr_x = out_x = 0
#             while curr_x + f <= in_dim:
#                 out[curr_f, out_y, out_x] = np.sum(filt[curr_f] * image[:,curr_y:curr_y+f, curr_x:curr_x+f]) + bias[curr_f]
#                 curr_x += s
#                 out_x += 1
#             curr_y += s
#             out_y += 1

# hello = np.array(array2).reshape(464,696)
# print(hello)














#show it
# cv2.imshow("some window", hello)
# cv2.waitKey(0)

# img = np.uint8(np.random.random((464, 696,3))*255)
# dst = cv2.integral(img)

# cv2.imshow("some window", img)
# cv2.waitKey(0)