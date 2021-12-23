
import numpy as np
import matplotlib.pyplot as plt

from _np_fix import *
from Datasets import *

init()
x, y = spiral_data(samples=100, classes=2)

plt.scatter(x[:,0], x[:,1], c=y, cmap='brg')
plt.axis('equal')
plt.show()