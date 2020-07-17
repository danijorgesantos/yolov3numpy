
import numpy as np

# inputs image with batch_size 416, 416, 32

#anchor-box

# pc probability that there is an object , between 1 and 0
# b coordenates of the box
# c different classes

# y = [ pc , bx, by, bh, bw, c1, c2, c3, pc , bx, by, bh, bw, c1, c2, c3  ]

# Briefly speaking, and I'll be using the example from the paper, for S=7, B=2 and C=20,
#  our output is a 7x7x30 tensor that encodes where (bounding box coordinates) and what the objects 
#  (probability of class) are. To achieve this, we construct a fully-connected layer at the end of our CNN 
#  that will give us 7x7x30 (rather forcefully). Hence on our first forward pass, each cell will have 2 random 
#  bounding boxes. A loss is calculated. The weights of the CNN will then be adjusted according to reduce that 
#  loss (opitimisation). Then the following passes will produce bounding boxes closer to the ground truth.










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


from butils import *


f1 = initializeFilter(10,)

w1 = initializeWeight(10)

# print(f1)

# print(w1)

b1 = np.zeros((f1.shape[0],1))
b2 = np.zeros((f1.shape[0],1))
b3 = np.zeros((w1.shape[0],1))
b4 = np.zeros((w1.shape[0],1))

params = [f1, w1, b1, b2, b3, b4]

print(params)


