from butils import *


f1 = initializeFilter(10,)

w1 = initializeWeight(10)

print(f1)
print(w1)

b1 = np.zeros((f1.shape[0],1))
b2 = np.zeros((f1.shape[0],1))
b3 = np.zeros((w1.shape[0],1))
b4 = np.zeros((w1.shape[0],1))

params = [f1, w1, b1, b2, b3, b4]