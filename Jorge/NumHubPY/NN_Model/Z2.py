import numpy as np
from _np_fix import init

from Datasets import *
from Layer import *
from ActvFunc import *
from Loss import *
from Optimizer import *
from Model import *
from Accuracy import *

init()
x, y = spiral_data(100, 2)
x_test, y_test = spiral_data(100, 2)
y = y.reshape(-1,1)
y_test = y_test.reshape(-1,1)

model = Model()

model.add(Layer_Dense(2,64,0,5e-4,0,5e-4))
model.add(Activation_ReLU())
model.add(Layer_Dense(64,1))
model.add(Activation_Sigmoid())

print(model.layers)

model.set(Loss_Binary_Cross_Entropy(), Optimizer_Adam(0.02, 5e-7), Accuracy_Categorical())

model.finalize()

model.train(x, y, 10000, 100, (x_test, y_test))
