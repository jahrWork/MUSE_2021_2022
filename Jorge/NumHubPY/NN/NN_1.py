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

lowest_loss = 1e9

for i in range(10001):
    d1.w += 0.05*np.random.randn(2,3)
    d1.b  += 0.05*np.random.randn(1,3)
    d2.w += 0.05*np.random.randn(3,3)
    d2.b  += 0.05*np.random.randn(1,3)

    d1.forward(x)
    f1.forward(d1.out)

    d2.forward(f1.out)
    f2.forward(d2.out)
    loss = L.calculate(f2.out, y)
    pred = np.argmax(f2.out, axis=1)
    acc = np.mean(pred == y)

    if loss < lowest_loss:
        print('New w set: ', i, 'loss: ', loss, 'acc', acc)
        bt_d1_w = d1.w.copy()
        bt_d1_b  = d1.b.copy()
        bt_d2_w = d2.w.copy()
        bt_d2_b  = d2.b.copy()
        lowest_loss = loss
    else:
        d1.w = bt_d1_w.copy()
        d1.b  = bt_d1_b.copy()
        d2.w = bt_d2_w.copy()
        d2.b  = bt_d2_b.copy()

test_nn1(d1, d2, f1, f2, L)