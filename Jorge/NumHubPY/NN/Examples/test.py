import numpy as np
import nnfs

from Layer import *
from ActvFunc import *
from Loss import *

nnfs.init()

out = np.array([[0.7, 0.1, 0.2],[0.1, 0.5, 0.4],[0.02, 0.9, 0.08]])
target = np.array([0, 1, 1])

sm_loss = Activation_Softmax_Loss_Categorical_Cross_Entropy()
sm_loss.backward(out, target)
dv1 = sm_loss.dinputs

act = Activation_Softmax()
act.output = out
loss = Loss_Categorical_Cross_Entropy()
loss.backward(out, target)
act.backward(loss.dinputs)
dv2 = act.dinputs

print(dv1)
print(dv2)