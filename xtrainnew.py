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




print('-------------------------------------------------------------')
print('Step 1. ----- all images in the array and ready to use *')
imgArray = []

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

#stride
conv_s = 1



#f1, f2, w3, w4 = (num_filt1 ,img_depth,f,f), (num_filt2 ,num_filt1,f,f), (128,800), (10, 128)

#filters
f1 = initializeFilter(8)
f2 = initializeFilter(8)

#weights
w3 = initializeWeight(8)
w4 = initializeWeight(8)

#bias
b1 = np.zeros((f1.shape[0],1))
b2 = np.zeros((f2.shape[0],1))
b3 = np.zeros((w3.shape[0],1))
b4 = np.zeros((w4.shape[0],1))


print(f1)
print(f2)





print('-------------------------------------------------------------')
print('Step 4. ----- do the foward propagation --> convolutions, and pooling *')


t = tqdm(imgArray, leave=True)

for image in t:
    '''
    Make predictions with trained filters/weights. 
    '''
    conv1 = convolution(image, f1, b1, conv_s) # convolution operation
    conv1[conv1<=0] = 0 #relu activation





















    
    t.set_description("Gen nº %i")
