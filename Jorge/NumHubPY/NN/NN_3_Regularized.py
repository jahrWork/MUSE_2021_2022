import numpy as np
from _np_fix import init

from Datasets import *
from Layer import *
from ActvFunc import *
from Loss import *
from Optimizer import *
from test_NN import *

init()
x, y = vertical_data(100, 3)

d1 = Layer_Dense(2, 64, 0, 5e-4, 0, 5e-4)
f1 = Activation_ReLU()

d2 = Layer_Dense(64,3)
lf2 = Activation_Softmax_Loss_Categorical_Cross_Entropy()
#opt = Optimizer_SGD(1., 1e-3, 0.9)
#opt = Optimizer_Adagrad(1, 1e-4, 1e-7)
#opt = Optimizer_RMS(0.02, 1e-5, 1e-7, 0.999)
opt = Optimizer_Adam(0.02, 5e-7, 1e-7, 0.9, 0.999)

for i in range(61):
    d1.forward(x)
    f1.forward(d1.out)

    d2.forward(f1.out)
    data_loss = lf2.forward(d2.out, y)

    reg_loss = lf2.loss.reg_loss(d1) + lf2.loss.reg_loss(d2)
    loss = reg_loss + data_loss

    pred = np.argmax(lf2.out, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    acc = np.mean(pred == y)

    if not i % 100:
        print(f'iteration: {i}, ' + f'acc: {acc:.3f}, ' + f'loss: {loss:.3f}, ' + f'data_loss: {data_loss:.3f}, ' + f'reg_loss: {reg_loss:.3f}, ' + f'lr: {opt.last_l_r}')

    lf2.backward(lf2.out, y)
    d2.backward(lf2.dins)
    f1.backward(d2.dins)
    d1.backward(f1.dins)

    opt.pre_update_params()
    opt.update_params(d1)
    opt.update_params(d2)
    opt.post_update_params()

test_nn(d1, d2, f1, lf2)