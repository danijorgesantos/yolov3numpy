
class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],4) 
        self.weights2   = np.random.rand(4,1)                 
        self.y          = y
        self.output     = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1)+b1)
        self.output = sigmoid(np.dot(self.layer1, self.weights2)+b2)

    def backprop(self):
        
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2



# import matplotlib.pyplot as plt 
# import numpy as np 
# import math 
  

# beta = 1  
# x = np.linspace(-10, 10, 100) 
# z = (x * (1/(1 + np.exp(beta * -x)))) 
  
# plt.plot(x, z) 
# plt.xlabel("x") 
# plt.ylabel("Sigmoid(X)") 
  
# plt.show() 






# beta = 1

# def swish(x, beta = 1):
#     return(x * sigmoid(beta * x))