import numpy as np
layer_outputs = np.array([4.8, 1.21, 2.385])
exp_values = np.exp(layer_outputs)
print(exp_values)
norm_values = exp_values/np.sum(exp_values)
print(norm_values)
print(sum(norm_values))

class Activation_Softmax:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

# nyu deep learning sp21 yann lecun
# fast.ai