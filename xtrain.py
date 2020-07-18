
import cv2
import numpy as np
from butils import *

import gzip


def extract_data(filename, num_images, IMAGE_WIDTH):
    '''
    Extract images by reading the file bytestream. Reshape the read values into a 3D matrix of dimensions [m, h, w], where m 
    is the number of training examples.
    '''
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_WIDTH * IMAGE_WIDTH * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, IMAGE_WIDTH*IMAGE_WIDTH)
        print(data[0])
        return data


def extract_labels(filename, num_images):
    '''
    Extract label into vector of integer values of dimensions [m, 1], where m is the number of images.
    '''
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels


# Get test data
m =10000
X = extract_data('t10k-images-idx3-ubyte.gz', m, 28)

# Normalize the data
X-= int(np.mean(X)) # subtract mean
X/= int(np.std(X)) # divide by standard deviation
test_data = np.hstack((X,y_dash))



X = test_data[:,0:-1]
X = X.reshape(len(test_data), 1, 28, 28)
y = test_data[:,-1]

corr = 0
digit_count = [0 for i in range(10)]
digit_correct = [0 for i in range(10)]

print()
print("Computing accuracy over test set:")

t = tqdm(range(len(X)), leave=True)















f1 = initializeFilter(10,)

w1 = initializeWeight(10)

print(f1)
print(w1)

b1 = np.zeros((f1.shape[0],1))
b2 = np.zeros((f1.shape[0],1))
b3 = np.zeros((w1.shape[0],1))
b4 = np.zeros((w1.shape[0],1))

params = [f1, w1, b1, b2, b3, b4]










# convolution function

def convolution(image, filt, bias, s=1):
    '''
    Confolves `filt` over `image` using stride `s`
    '''
    (n_f, n_c_f, f, _) = filt.shape # filter dimensions
    n_c, in_dim, _ = image.shape # image dimensions
    
    out_dim = int((in_dim - f)/s)+1 # calculate output dimensions
    
    assert n_c == n_c_f, "Dimensions of filter must match dimensions of input image"
    
    out = np.zeros((n_f,out_dim,out_dim))
    
    # convolve the filter over every part of the image, adding the bias at each step. 
    for curr_f in range(n_f):
        curr_y = out_y = 0
        while curr_y + f <= in_dim:
            curr_x = out_x = 0
            while curr_x + f <= in_dim:
                out[curr_f, out_y, out_x] = np.sum(filt[curr_f] * image[:,curr_y:curr_y+f, curr_x:curr_x+f]) + bias[curr_f]
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
        
    return out







# first convolution operation

conv1 = convolution(image, f1, b1, conv_s) # convolution operation
conv1[conv1<=0] = 0 #relu activation










pc = 1
bx = 150
by = 150
bh = 550
bw = 370
c1 = 0  # pedestrian
c2 = 1  # car
c3 = 0  # traffic light


y = [pc, bx, by, bh, bw, c1, c2, c3]

print(y)

# Python program to explain cv2.rectangle() method  
   
# Reading an image in default mode 
image = cv2.imread('mer.png') 
   
# Window name in which image is displayed 
window_name = 'Image'
  
# Start coordinate, here (5, 5) 
# represents the top left corner of rectangle 
start_point = (bx, by) 
  
# Ending coordinate, here (220, 220) 
# represents the bottom right corner of rectangle 
end_point = (bh, bw) 
  
# Blue color in BGR 
color = (255, 0, 0) 
  
# Line thickness of 2 px 
thickness = 2
  
# Using cv2.rectangle() method 
# Draw a rectangle with blue line borders of thickness of 2 px 
image = cv2.rectangle(image, start_point, end_point, color, thickness) 
  
# Displaying the image  
cv2.imshow(window_name, image)  
cv2.waitKey(0)
cv2.destroyAllWindows()


