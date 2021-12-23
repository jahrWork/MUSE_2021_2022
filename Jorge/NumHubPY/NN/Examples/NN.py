import numpy as np

inputs = np.array([1, 2, 3, 4.])
weights = np.array([0.5, 0.8, -0.3, 1.8])
bias = 2.9

output = np.dot(weights,inputs)+bias
print(output)