import cv2
import numpy as np
from os.path import isfile, join

mypath = 'train'

img = cv2.imread(join(mypath, 'Toyota-Yaris-01-696x464.jpg'),0)
rows,cols = img.shape

print(img.shape)

array = []

array.append(img)

print(array)

for i in range(rows):
    for j in range(cols):
        k = img[i,j]