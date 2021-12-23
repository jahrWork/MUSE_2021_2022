import numpy as np
from Loss import *
"""
ins = inputs
out = output
v   = values
y   = lables
"""
class Activation_ReLU:

    def forward(self, ins):
        self.ins = ins
        self.out = np.maximum(0, ins)

    def backward(self, dv):
        self.dins = dv.copy()
        self.dins[self.ins <= 0] = 0

    def predictions(self, out):
        return out


class Activation_Softmax:

    def forward(self, ins):
        self.ins = ins
        exp_v = np.exp(ins - np.max(ins, axis=1, keepdims=True))
        prob = exp_v/np.sum(exp_v, axis=1, keepdims=True) 
        self.out = prob

    def backward(self, dv):
        self.dins = np.empty_like(dv)
        for i, (single_out, single_dv) in enumerate(zip(self.out, dv)):
            single_out = single_out.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_out) - np.dot(single_out, single_out.T)
            self.dins[i] = np.dot(jacobian_matrix, single_dv)

    def predictions(self, out):
        return np.argmax(out, axis=1)


class Activation_Softmax_Loss_Categorical_Cross_Entropy():

    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_Categorical_Cross_Entropy()

    def forward(self, ins, y_true):
        self.activation.forward(ins)
        self.out = self.activation.out
        return self.loss.calculate(self.out, y_true)

    def backward(self, dv, y_true):
        samples = len(dv)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dins = dv.copy()
        self.dins[range(samples), y_true] -= 1
        self.dins = self.dins/samples


class Activation_Sigmoid:

    def forward(self, ins):
        self.ins = ins
        self.out = 1/(1 + np.exp(-ins))

    def backward(self, dv):
        self.dins = dv*(1 - self.out)*self.out

    def predictions(self, out):
        return (out > 0.5)*1


class Activation_Linear:

    def forward(self, ins):
        self.ins = ins
        self.out = ins

    def backward(self, dv):
        self.dins = dv.copy()

    def predictions(self, out):
        return out