
import numpy as np

inputs = np.array([1, 2, 3, 2.5])
weights = np.array([[0.2, 0.8, -0.5, 1],[0.5, -0.91, 0.26, -0.5],[0.2, 0.8, -0.5, 1]]).T
bias = np.array([2, 3, 0.5])

output = np.dot(inputs,weights)+bias
print(output)