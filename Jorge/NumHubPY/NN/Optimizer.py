import numpy as np
"""
l_r = learning rate
dc  = decay
m   = momentum
w   = weight
b   = bias
dv  = dvalues
i   = iteration
"""
class Optimizer_SGD:

    def __init__(self, l_r=1., dc=0., m=0.):
        self.l_r = l_r
        self.last_l_r = l_r
        self.dc = dc
        self.i = 0
        self.m = m

    def pre_update_params(self):
        if self.dc:
            self.last_l_r = self.l_r*(1./(1. + self.dc*self.i))

    def update_params(self, layer):
        if self.m:
            if not hasattr(layer, 'w_m'):
                layer.w_m = np.zeros_like(layer.w)
                layer.b_m = np.zeros_like(layer.b)

            w_updates = self.m*layer.w_m - self.last_l_r*layer.dw
            b_updates = self.m*layer.b_m - self.last_l_r*layer.db
            
            layer.w_m = w_updates
            layer.b_m = b_updates

        else:
            w_updates = -self.last_l_r*layer.dw
            b_updates = -self.last_l_r*layer.db
        layer.w += w_updates
        layer.b += b_updates

    def post_update_params(self):
        self.i += 1


class Optimizer_Adagrad:

    def __init__(self, l_r=1.0, dc=0., eps=1e-7):
        self.l_r = l_r
        self.last_l_r = l_r
        self.dc = dc
        self.i = 0
        self.eps = eps

    def pre_update_params(self):
        if self.dc:
            self.last_l_r = self.l_r*(1./(1. + self.dc*self.i))

    def update_params(self, layer):
        if not hasattr(layer, 'w_cache'):
            layer.w_cache = np.zeros_like(layer.w)
            layer.b_cache = np.zeros_like(layer.b)
        layer.w_cache += layer.dw**2
        layer.b_cache += layer.db**2

        layer.w += - self.last_l_r*layer.dw/(np.sqrt(layer.w_cache) + self.eps)
        layer.b += - self.last_l_r*layer.db/(np.sqrt(layer.b_cache) + self.eps)

    def post_update_params(self):
        self.i += 1


class Optimizer_RMS:

    def __init__(self, l_r=0.001, dc=0., eps=1e-7, rho=0.9):
        self.l_r = l_r
        self.last_l_r = l_r
        self.dc = dc
        self.i = 0
        self.eps = eps
        self.rho = rho

    def pre_update_params(self):
        if self.dc:
            self.last_l_r = self.l_r*(1./(1. + self.dc*self.i))

    def update_params(self, layer):
        if not hasattr(layer, 'w_cache'):
            layer.w_cache = np.zeros_like(layer.w)
            layer.b_cache   = np.zeros_like(layer.b)
        layer.w_cache = self.rho*layer.w_cache + (1 - self.rho)*layer.dw**2
        layer.b_cache = self.rho*layer.b_cache + (1 - self.rho)*layer.db**2

        layer.w += - self.last_l_r*layer.dw/(np.sqrt(layer.w_cache) + self.eps)
        layer.b += - self.last_l_r*layer.db/(np.sqrt(layer.b_cache) + self.eps)

    def post_update_params(self):
        self.i += 1


class Optimizer_Adam:

    def __init__(self, l_r=0.001, dc=0., eps=1e-7, beta_1=0.9, beta_2=0.999):
        self.l_r = l_r
        self.last_l_r = l_r
        self.dc = dc
        self.i = 0
        self.eps = eps
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        if self.dc:
            self.last_l_r = self.l_r*(1./(1. + self.dc*self.i))

    def update_params(self, layer):

        if not hasattr(layer, 'w_cache'):
            layer.w_ms = np.zeros_like(layer.w)
            layer.w_cache = np.zeros_like(layer.w)
            layer.b_ms = np.zeros_like(layer.b)
            layer.b_cache = np.zeros_like(layer.b)

        layer.w_ms = self.beta_1*layer.w_ms + (1 - self.beta_1)*layer.dw
        layer.b_ms = self.beta_1*layer.b_ms + (1 - self.beta_1)*layer.db

        w_ms_corrected = layer.w_ms/(1 - self.beta_1**(self.i + 1))
        b_ms_corrected = layer.b_ms/(1 - self.beta_1**(self.i + 1))

        layer.w_cache = self.beta_2*layer.w_cache + (1 - self.beta_2)*layer.dw**2
        layer.b_cache = self.beta_2*layer.b_cache + (1 - self.beta_2)*layer.db**2

        w_cache_corrected = layer.w_cache/(1 - self.beta_2**(self.i + 1))
        b_cache_corrected = layer.b_cache/(1 - self.beta_2**(self.i + 1))

        layer.w += -self.last_l_r*w_ms_corrected/(np.sqrt(w_cache_corrected) + self.eps)
        layer.b += -self.last_l_r*b_ms_corrected/(np.sqrt(b_cache_corrected) + self.eps)

    def post_update_params(self):
        self.i += 1