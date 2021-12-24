import numpy as np
from Datasets import *
from _np_fix import *

init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights=0.01*np.random.randn(n_inputs, n_neurons)
        self.biases=np.zeros((1,n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


x, y = spiral_data(samples=100, classes=3)
dense1 = Layer_Dense(2,3)
dense1.forward(x)

print(dense1.output)