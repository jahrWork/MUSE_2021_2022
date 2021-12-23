import numpy as np
import matplotlib.pyplot as plt

x = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 4.25, 4.5, 4.75, 5, 5.5]
y = [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1]
s = np.arange(0.5, 5.5, 0.0025)
plt.plot(x, y, '*')
ii = 0 - 1
LL = np.zeros([1001,1001])
for i in np.arange(-5, 5.01, 0.01):
    ii = ii + 1
    jj = 0 - 1
    print(1001 - ii)
    for j in np.arange(-5, 5.01, 0.01):
        jj = jj + 1
        p = [i,j]
        def f(x1):
           return p[0]*x1 + p[1]
        def g(s):
           return 1/(np.exp(-f(s)) + 1)
        for L in range(np.size(x)):
            if y[L] == 1:
                LL[ii,jj] = LL[ii,jj] + np.log(g(x[L]))
            else:
                LL[ii,jj] = LL[ii,jj] + np.log(1 - g(x[L]))
LLL = np.max(np.max(LL))
row = np.where(LL == LLL)[0]
col = np.where(LL == LLL)[1]
p[0] = row/100 - 5
p[1] = col/100 - 5
def f(x1):
   return p[0]*x1 + p[1]
def g(s):
   return 1/(np.exp(-f(s)) + 1)
plt.plot(s, f(s))
plt.plot(s, g(s))
plt.ylim([0,1])
plt.show()