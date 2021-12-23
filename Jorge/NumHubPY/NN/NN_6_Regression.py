import numpy as np
from _np_fix import init

from Datasets import *
from Layer import *
from ActvFunc import *
from Loss import *
from Optimizer import *
from test_NN import *

init()
x, y = sine_data(1000)

d1 = Layer_Dense(1, 64)
f1 = Activation_ReLU()

d2 = Layer_Dense(64,1)
f2 = Activation_Linear()

L = Loss_MSE()

#opt = Optimizer_SGD(1., 1e-3, 0.9)
#opt = Optimizer_Adagrad(1, 1e-4, 1e-7)
#opt = Optimizer_RMS(0.02, 1e-5, 1e-7, 0.999)
opt = Optimizer_Adam()

acc_precision = np.std(y)/250

for i in range(10001):
    d1.forward(x)
    f1.forward(d1.out)

    d2.forward(f1.out)
    f2.forward(d2.out)
    data_loss = L.calculate(f2.out, y)

    reg_loss = L.reg_loss(d1) + L.reg_loss(d2)
    loss = reg_loss + data_loss

    pred = f2.out
    acc = np.mean(np.absolute(pred - y) < acc_precision)

    if not i % 100:
        print(f'iteration: {i}, ' + f'acc: {acc:.3f}, ' + f'loss: {loss:.3f}, ' + f'data_loss: {data_loss:.3f}, ' + f'reg_loss: {reg_loss:.3f}, ' + f'lr: {opt.last_l_r}')
 
    L.backward(f2.out, y)
    f2.backward(L.dins)
    d2.backward(f2.dins)
    f1.backward(d2.dins)
    d1.backward(f1.dins)

    opt.pre_update_params()
    opt.update_params(d1)
    opt.update_params(d2)
    opt.post_update_params()

test_nn6(d1, d2, f1, f2, L, acc_precision)