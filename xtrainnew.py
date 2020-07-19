import argparse
import gzip
import pickle
from os import listdir
from os.path import isfile, join
import skimage.measure

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
    image = cv2.imread(join(mypath, img),0)
    imgArray.append(image)

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

pool_f = 2
pool_s = 2

#filters
f1 = initializeFilter(25).reshape(5,5)
f2 = initializeFilter(25).reshape(5,5)
f3 = initializeFilter(25).reshape(5,5)
f4 = initializeFilter(25).reshape(5,5)
f5 = initializeFilter(25).reshape(5,5)
f6 = initializeFilter(25).reshape(5,5)

#weights
w3 = initializeWeight(275)
w4 = initializeWeight(275)

#bias
b1 = np.zeros((f1.shape[0],1))
b2 = np.zeros((f2.shape[0],1))
b3 = np.zeros((w3.shape[0],1))
b4 = np.zeros((w4.shape[0],1))


print('-------------------------------------------------------------')
print('Step 4. ----- do the foward propagation --> convolutions, and pooling *')


t = tqdm(imgArray, leave=True)

for image in t:
    '''
    Make predictions with trained filters/weights. 
    '''

    # convolution operation
    conv1 = cv2.filter2D(image,-1,f1)
    conv1[conv1<=0] = 0 #relu activation

    # #show img
    # cv2.imshow("some window", conv1)
    # cv2.waitKey(0)

    # maxpooling operation
    pooled = skimage.measure.block_reduce(conv1, (2,2), np.max) 

    # #show img
    # cv2.imshow("some window", pooled)
    # cv2.waitKey(0)

    # convolution operation 2
    conv2 = cv2.filter2D(pooled,-1,f2)
    conv2[conv2<=0] = 0 #relu activation

    # #show img
    # cv2.imshow("some window", conv2)
    # cv2.waitKey(0)

    # maxpooling operation
    pooled2 = skimage.measure.block_reduce(conv2, (2,2), np.max) 

    # #show img
    # cv2.imshow("some window", pooled2)
    # cv2.waitKey(0)

    # convolution operation 3
    conv3 = cv2.filter2D(pooled2,-1,f3)
    conv3[conv3<=0] = 0 #relu activation

    # #show img
    # cv2.imshow("some window", conv3)
    # cv2.waitKey(0)

    # maxpooling operation
    pooled3 = skimage.measure.block_reduce(conv3, (2,2), np.max) 

    # #show img
    # cv2.imshow("some window", pooled3)
    # cv2.waitKey(0)

    # convolution operation 4
    conv4 = cv2.filter2D(pooled3,-1,f4)
    conv4[conv4<=0] = 0 #relu activation

    # #show img
    # cv2.imshow("some window", conv4)
    # cv2.waitKey(0)

    # maxpooling operation
    pooled4 = skimage.measure.block_reduce(conv4, (2,2), np.max) 

    # #show img
    # cv2.imshow("some window", pooled4)
    # cv2.waitKey(0)

    # convolution operation 5
    conv5 = cv2.filter2D(pooled4,-1,f5)
    conv5[conv5<=0] = 0 #relu activation

    # #show img
    # cv2.imshow("some window", conv5)
    # cv2.waitKey(0)

    # maxpooling operation
    pooled5 = skimage.measure.block_reduce(conv5, (2,2), np.max) 

    # #show img
    # cv2.imshow("some window", pooled5)
    # cv2.waitKey(0)

    # convolution operation 6
    conv6 = cv2.filter2D(pooled5,-1,f6)
    conv6[conv6<=0] = 0 #relu activation

    # #show img
    # cv2.imshow("some window", conv6)
    # cv2.waitKey(0)

    # maxpooling operation
    pooled6 = skimage.measure.block_reduce(conv6, (2,2), np.max) 

    # #show img
    # cv2.imshow("some window", pooled6)
    # cv2.waitKey(0)
    
    # flaten pooled6
    (nf2, dim2) = pooled6.shape
    fc = pooled6.reshape((nf2 * dim2, 1)) # flatten pooled layer

    z = w3.dot(fc) + b3 # first dense layer
    z[z<=0] = 0 # pass through ReLU non-linearity
    
    out = w4.dot(z) + b4 # second dense layer
    probs = softmax(out) # predict class probabilities with the softmax activation function


    bla = np.argmax(probs)
    bla2 = np.max(probs)




    # description for the for loop
    t.set_description("Gen nº %i")
