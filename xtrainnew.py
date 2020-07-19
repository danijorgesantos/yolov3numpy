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

    # image = plt.imread(img)
    # im = image

    # im = Image.open( join(mypath, img), "r")
    # pix_val = list(im.getdata())
    # pix_val_flat = [x for sets in pix_val for x in sets]

    image = cv2.imread(join(mypath, img),0)
    imgArray.append(image)

print(imgArray)

print('-------------------------------------------------------------')
print('Step 2. ----- all labels in the array and ready to use *')




print('-------------------------------------------------------------')
print('Step 3. ----- initializing the weights, filters and parameters *')
print('-------------------------------------------------------------')

#stride
conv_s = 1

num_classes = 10
lr = 0.01
beta1 = 0.95
beta2 = 0.99
img_dim = 28
img_depth = 1
f = 5
num_filt1 = 8
num_filt2 = 8
batch_size = 32
num_epochs = 2

f1, f2, w3, w4 = (num_filt1 ,img_depth,f,f), (num_filt2 ,num_filt1,f,f), (128,800), (10, 128)

#filters
f1 = initializeFilter(f1)
f2 = initializeFilter(f2)

#weights
w3 = initializeWeight(w3)
w4 = initializeWeight(w4)

#bias
b1 = np.zeros((f1.shape[0],1))
b2 = np.zeros((f2.shape[0],1))
b3 = np.zeros((w3.shape[0],1))
b4 = np.zeros((w4.shape[0],1))


# print(f1.shape)
# print(f2)





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
