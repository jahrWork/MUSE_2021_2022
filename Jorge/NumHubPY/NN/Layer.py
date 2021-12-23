import numpy as np
"""
w   = weight
b   = bias
reg = regularization
ins = inputs
out = output
dv  = dvalues
"""
class Layer_Dense:

    def __init__(self, n_ins, n_neurons, w_reg_l1=0, w_reg_l2=0, b_reg_l1=0, b_reg_l2=0):
        self.w = 0.05*np.random.randn(n_ins, n_neurons)
        self.b = np.zeros((1,n_neurons))
        
        self.w_reg_l1 = w_reg_l1
        self.w_reg_l2 = w_reg_l2
        self.b_reg_l1 = b_reg_l1
        self.b_reg_l2 = b_reg_l2

    def forward(self, ins):
        self.ins = ins
        self.out = np.dot(ins, self.w) + self.b
    
    def backward(self, dv):
        self.dw = np.dot(self.ins.T, dv)
        self.db = np.sum(dv, axis=0, keepdims=True)

        if self.w_reg_l1 > 0:
            dL1 = np.ones_like(self.w)
            dL1[self.w < 0] = -1
            self.dw += self.w_reg_l1*dL1

        if self.w_reg_l2 > 0:
            self.dw += 2*self.w_reg_l2*self.w

        if self.b_reg_l1   > 0:
            dL1 = np.ones_like(self.b)
            dL1[self.b < 0] = -1
            self.db += self.b_reg_l1*dL1

        if self.b_reg_l2   > 0:
            self.db  += 2*self.b_reg_l2*self.b
        self.dins = np.dot(dv, self.w.T)


class Layer_Dropout:

    def __init__(self, rate):
        self.rate = 1- rate

    def forward(self, ins):
        self.ins = ins
        self.binary_mask = np.random.binomial(1, self.rate, size=ins.shape)/self.rate
        self.out = ins*self.binary_mask
    
    def backward(self, dv):
        self.dins = dv * self.binary_mask