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
x, y = sine_data()

model = Model()

model.add(Layer_Dense(1,64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64,64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64,1))
model.add(Activation_Linear())

print(model.layers)

model.set(Loss_MSE(), Optimizer_Adam(0.005, 1e-3), Accuracy_Regression())

model.finalize()

model.train(x, y, 10000, 100)