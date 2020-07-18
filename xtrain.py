
import argparse
import gzip
import pickle
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

from butils import *
from bforward import *




## Initializing all the parameters, filters, weights ---------------------------------------------------------------------------------------------------

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
save_path = 'params.pkl'

conv_s = 1

f1, f2, w3, w4 = (num_filt1 ,img_depth,f,f), (num_filt2 ,num_filt1,f,f), (128,800), (10, 128)

f1 = initializeFilter(f1)
f2 = initializeFilter(f2)
w3 = initializeWeight(w3)
w4 = initializeWeight(w4)

b1 = np.zeros((f1.shape[0],1))
b2 = np.zeros((f2.shape[0],1))
b3 = np.zeros((w3.shape[0],1))
b4 = np.zeros((w4.shape[0],1))

params = [f1, f2, w3, w4, b1, b2, b3, b4]

cost = []

# Get test data -------------------------------------------------------------------------------------------------------------------------------------------

m = 2
X = extract_data('sample.tar', m, 28)
#y_dash = extract_labels('t10k-labels-idx1-ubyte.gz', m).reshape(m,1)
y_dash = [[1],[2]]
# Normalize the data
X-= int(np.mean(X)) # subtract mean
X/= int(np.std(X)) # divide by standard deviation
test_data = np.hstack((X,y_dash))

X = test_data[:,0:-1]
X = X.reshape(len(test_data), 1, 28, 28)
# y = test_data[:,-1]

corr = 0
digit_count = [0 for i in range(10)]
digit_correct = [0 for i in range(10)]

t = tqdm(range(len(X)), leave=True)

for i in t:
    x = X[i]
    conv1 = convolution(x, f1, b1, conv_s) # convolution operation
    conv1[conv1<=0] = 0 #relu activation


    # conv2 = convolution(conv1, f2, b2, conv_s) # second convolution operation
    # conv2[conv2<=0] = 0 # pass through ReLU non-linearity
    # pooled = maxpool(conv2, pool_f, pool_s) # maxpooling operation
    # (nf2, dim2, _) = pooled.shape
    # fc = pooled.reshape((nf2 * dim2 * dim2, 1)) # flatten pooled layer
    # z = w3.dot(fc) + b3 # first dense layer
    # z[z<=0] = 0 # pass through ReLU non-linearity
    
# out = w4.dot(z) + b4 # second dense layer
# probs = softmax(out) # predict class probabilities with the softmax activation function

# digit_count[int(y[i])]+=1
# if pred==y[i]:
#     corr+=1
#     digit_correct[pred]+=1

t.set_description("Acc:%0.2f%%" % (float(corr/(i+1))*100))

print(t)















# f1 = initializeFilter(10,)

# w1 = initializeWeight(10)

# print(f1)
# print(w1)

# b1 = np.zeros((f1.shape[0],1))
# b2 = np.zeros((f1.shape[0],1))
# b3 = np.zeros((w1.shape[0],1))
# b4 = np.zeros((w1.shape[0],1))

# params = [f1, w1, b1, b2, b3, b4]










# # convolution function

# def convolution(image, filt, bias, s=1):
#     '''
#     Confolves `filt` over `image` using stride `s`
#     '''
#     (n_f, n_c_f, f, _) = filt.shape # filter dimensions
#     n_c, in_dim, _ = image.shape # image dimensions
    
#     out_dim = int((in_dim - f)/s)+1 # calculate output dimensions
    
#     assert n_c == n_c_f, "Dimensions of filter must match dimensions of input image"
    
#     out = np.zeros((n_f,out_dim,out_dim))
    
#     # convolve the filter over every part of the image, adding the bias at each step. 
#     for curr_f in range(n_f):
#         curr_y = out_y = 0
#         while curr_y + f <= in_dim:
#             curr_x = out_x = 0
#             while curr_x + f <= in_dim:
#                 out[curr_f, out_y, out_x] = np.sum(filt[curr_f] * image[:,curr_y:curr_y+f, curr_x:curr_x+f]) + bias[curr_f]
#                 curr_x += s
#                 out_x += 1
#             curr_y += s
#             out_y += 1
        
#     return out







# # first convolution operation

# conv1 = convolution(image, f1, b1, conv_s) # convolution operation
# conv1[conv1<=0] = 0 #relu activation










# pc = 1
# bx = 150
# by = 150
# bh = 550
# bw = 370
# c1 = 0  # pedestrian
# c2 = 1  # car
# c3 = 0  # traffic light


# y = [pc, bx, by, bh, bw, c1, c2, c3]

# print(y)

# # Python program to explain cv2.rectangle() method  
   
# # Reading an image in default mode 
# image = cv2.imread('mer.png') 
   
# # Window name in which image is displayed 
# window_name = 'Image'
  
# # Start coordinate, here (5, 5) 
# # represents the top left corner of rectangle 
# start_point = (bx, by) 
  
# # Ending coordinate, here (220, 220) 
# # represents the bottom right corner of rectangle 
# end_point = (bh, bw) 
  
# # Blue color in BGR 
# color = (255, 0, 0) 
  
# # Line thickness of 2 px 
# thickness = 2
  
# # Using cv2.rectangle() method 
# # Draw a rectangle with blue line borders of thickness of 2 px 
# image = cv2.rectangle(image, start_point, end_point, color, thickness) 
  
# # Displaying the image  
# cv2.imshow(window_name, image)  
# cv2.waitKey(0)
# cv2.destroyAllWindows()
