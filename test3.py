import numpy as np

# input layer
input = np.array([[1, 2, 3, 4]])

# initialize bias, betas and wieghts --------------
b1 = 2
b2 = 1

beta2 = 1
beta = 1

fcweighs1 = np.random.rand(input.shape[1],16)

# --------------------------------------------------

print('--------------------')
print('wights',fcweighs1)

# feed foward layer 1 with activation function swish
x = np.dot(input, fcweighs1)+b1
layer1Result = (x * (1/(1 + np.exp(beta * -x))))

print('----------------------------')
print('layer 1 result --> ', layer1Result)

fcweighs2 = np.random.rand(layer1Result.shape[1],16) 

# feed foward layer 2 with activation function swish
x = np.dot(layer1Result, fcweighs2)+b1
layer2Result = (x * (1/(1 + np.exp(beta * -x))))

# feed foward layer 2 with activation function softmax for final layer


print('----------------------------')
print('layer 2 result --> ', layer2Result)

