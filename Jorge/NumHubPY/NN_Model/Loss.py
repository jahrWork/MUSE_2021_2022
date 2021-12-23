import numpy as np
from numpy.core.numeric import outer
from ActvFunc import *
"""
l_r = learning rate
dc  = decay
m   = momentum
w   = weight
b   = bias
dv  = dvalues
i   = iteration
reg = regularization
y   = outcome
ins = inputs
out = output
"""
class Loss:

    def reg_loss(self):
        reg_loss = 0
        for layer in self.trainable_layers:
            if layer.w_reg_l1 > 0:
                reg_loss += layer.w_reg_l1*np.sum(np.abs(layer.w))
            if layer.w_reg_l2 > 0:
                reg_loss += layer.w_reg_l2*np.sum(np.abs(layer.w*layer.w))

            if layer.b_reg_l1 > 0:
                reg_loss += layer.b_reg_l1*np.sum(np.abs(layer.b))
            if layer.b_reg_l2 > 0:
                reg_loss += layer.b_reg_l1*np.sum(np.abs(layer.b*layer.b))
        return reg_loss

    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def calculate(self, out, y, include_reg=False):
        sample_losses = self.forward(out, y)
        data_loss = np.mean(sample_losses)
        if not include_reg:
            return data_loss
        return data_loss, self.reg_loss()


class Loss_Categorical_Cross_Entropy(Loss):

    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dv, y_true):
        samples = len(dv)
        labels = len(dv[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        self.dins = -y_true/dv
        self.dins = self.dins/samples


class Loss_Binary_Cross_Entropy(Loss):

    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        sample_losses = -(y_true*np.log(y_pred_clipped) + (1 - y_true)*np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)
        return sample_losses

    def backward(self, dv, y_true):
        samples = len(dv)
        outs = len(dv[0])
        clipped_dv = np.clip(dv, 1e-7, 1 - 1e-7)
        self.dins = -(y_true/clipped_dv - (1 - y_true)/(1 - clipped_dv))/outs
        self.dins = self.dins/samples


class Loss_MSE(Loss):

    def forward(self, y_pred, y_true):
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
        return sample_losses

    def backward(self, dv, y_true):
        samples = len(dv)
        out = len(dv[0])
        self.dins = -2*(y_true - dv)/out
        self.dins = self.dins/samples


class Loss_MAE(Loss):

    def forward(self, y_pred, y_true):
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)
        return sample_losses

    def backward(self, dv, y_true):
        samples = len(dv)
        out = len(dv[0])
        self.dins = np.sign(y_true - dv)/out
        self.dins = self.dins/samples