import cv2
import numpy as np
from os.path import isfile, join

mypath = 'train'

img = cv2.imread(join(mypath, 'Toyota-Yaris-01-696x464.jpg'),0)
rows,cols = img.shape


h_img, w_img = img.shape # image dimensions, h_img = 464, w_img = 696

def initializeFilter(size, scale = 1.0):
    stddev = scale/np.sqrt(np.prod(size))
    return np.random.normal(loc = 0, scale = stddev, size = size)

f1 = initializeFilter(9).reshape(3,3)

# print(f1)

h_filt, w_filt  = f1.shape # filter dimensions

# print(h_filt)
# print(w_filt)


#print(img[0])

rowArray = []
sectionArray  = []


for i in range(int(w_img/3)):
    for a in range(3):
        for i in range(3):
            rowArray.append(img[a][i])
        sectionArray.append(rowArray)
        rowArray = []
        print('number', i)

print(sectionArray)
print(f1)

square = []
for i in range(len(sectionArray)):
    square.append(np.multiply(sectionArray, f1))

print(square)

imgUint8 = np.uint8(square)

cv2.imshow("s", imgUint8)
cv2.waitKey(0)





array = []
array2= []

array.append(img)

#print(array)

for i in range(rows):
    for j in range(cols):
        k = img[i,j]
        array2.append(k)


# hello = np.array(array2).reshape(464,696)
# print(hello)


#show it
# cv2.imshow("some window", hello)
# cv2.waitKey(0)

# img = np.uint8(np.random.random((464, 696,3))*255)
# dst = cv2.integral(img)

# cv2.imshow("some window", img)
# cv2.waitKey(0)