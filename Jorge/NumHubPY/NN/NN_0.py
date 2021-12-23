import numpy as np
from _np_fix import init

from Datasets import *
from Layer import *
from ActvFunc import *
from Loss import *
from Optimizer import *
from test_NN import *

init()
x, y = spiral_data(100, 3)

d1 = Layer_Dense(2,3)
f1 = Activation_ReLU()

d2 = Layer_Dense(3,3)
f2 = Activation_Softmax()

L = Loss_Categorical_Cross_Entropy()


d1.forward(x)
f1.forward(d1.out)

d2.forward(f1.out)
f2.forward(d2.out)

loss = L.calculate(f2.out, y)

pred = np.argmax(f2.out, axis=1)
if len(y.shape) == 2:
   y = np.argmax(y, axis=1)
acc = np.mean(pred == y)

print(f2.out[:5])
print('loss: ',loss)
print('acc: ',acc)

test_nn1(d1, d2, f1, f2, L)