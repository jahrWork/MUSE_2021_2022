import numpy as np
import matplotlib.pyplot as plt

from Cauchy import *
from a_zeros import new2
from a_linalg import QR


mu = 0.012

def FFF(U,t):
    x, y, z, vx, vy, vz = U[0], U[1], U[2], U[3], U[4], U[5]
    d = np.sqrt((x + mu)**2 + y**2 + z**2)
    r = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)

    dvxdt = x + 2*vy - (1 - mu)*(x + mu)/d**3 - mu*(x - 1 + mu)/r**3
    dvydt = y - 2*vx - (1 - mu)*y/d**3 - mu*y/r**3
    dvzdt = -(1 - mu)*z/d**3 - mu*z/r**3

    return np.array([vx, vy, vz, dvxdt, dvydt, dvzdt])

def G(Y, t):
    X = np.zeros(6)
    X[0:3] = Y
    X[3:6] = 0.

    GX = FFF(X, 0.)
    return GX[3:6]

N = 20000
M = 6
U = np.zeros([N,M])
E = np.zeros([N,M])
t = np.zeros(N)
NL = 5
U0 = np.zeros([NL,M])
A = np.zeros([M,M])


t0 = 0
tf = 4*np.pi/0.3

t = np.linspace(t0,tf,N)

U0[0,:] = [0.8, 0.6, 0., 0., 0., 0.]
U0[1,:] = [0.8, -0.6, 0., 0., 0., 0.]
U0[2,:] = [-0.1, 0.0, 0., 0., 0., 0.]
U0[3,:] = [0.1, 0.0, 0., 0., 0., 0.]
U0[4,:] = [1.1, 0.0, 0., 0., 0., 0.]

for i in range(0, NL):
    U0[i,0:3] = new2(U0[i,0:3], G)
    A = Jc(FFF, U0[i,:])
    print('LP = ', U0[i,:])
    print('F in LP = ', FFF(U0[i,:], 0.))

    lm = QR(A)
    for j in range(M):
        print('EigV = ', lm[j])

    eps = np.random.randint(2,size=M)*1e-2
    U[0,:] = U0[i,:] + eps
    [th, U] = ca(N, tf, U[0,:], FFF, eval('rk4'))

    A = U[:,:3]
    B = U0[i,0:3]

    plt.plot(A[:,0]-B[0], A[:,1]-B[1])
    plt.show()