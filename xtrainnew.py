import argparse
import gzip
import pickle
from os import listdir
from os.path import isfile, join

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

from bforward import *
from butils import *

mypath = 'train'

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

print(onlyfiles)


imgArray = []


print('-------------------------------------------------------------')
print('Step 1. ----- all images in the array and ready to use *')

for img in tqdm(onlyfiles):
    im = Image.open( join(mypath, img), "r")
    pix_val = list(im.getdata())
    pix_val_flat = [x for sets in pix_val for x in sets]
    imgArray.append(pix_val_flat)

print('-------------------------------------------------------------')
print('Step 2. ----- all labels in the array and ready to use *')




print('-------------------------------------------------------------')
print('Step 3. ----- initializing the weights, filters and parameters *')
print('-------------------------------------------------------------')


def initializeFilter(size, scale = 1.0):
    stddev = scale/np.sqrt(np.prod(size))
    return np.random.normal(loc = 0, scale = stddev, size = size)

def initializeWeight(size):
    return np.random.standard_normal(size=size) * 0.01

f1 = initializeFilter(8)
f2 = initializeFilter(8)

print(f1)
print(f2)





print('-------------------------------------------------------------')
print('Step 4. ----- do the foward propagation --> convolutions, and pooling *')


t = tqdm(imgArray, leave=True)

for i in t:
    t.set_description("Gen nº %i")
