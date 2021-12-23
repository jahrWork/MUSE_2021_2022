import numpy as np
inputs = np.array([0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100])
output = np.maximum(0, inputs)
print(output)

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)