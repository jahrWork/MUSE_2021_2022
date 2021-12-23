import numpy as np
import matplotlib.pyplot as plt
from Sch_R import *
from ODE import *

def RA(n, nn, R):
    x, s, p1 = np.linspace(-n, n, nn), np.zeros([nn*nn,2]), 0
    for i in range(nn):
        for j in range(nn):
            if np.abs(R(x[i] + x[j]*complex(0,1))) <= 1:
                s[p1,:] = [x[i],x[j]]
                p1 += 1
    return s
sch_label = input('Scheme: ')
s = RA(3,801,eval(sch_label))
plt.scatter(s[:,0], s[:,1], s=4)
plt.axis('scaled')
#plt.axhline(y=0, color='k')
#plt.axvline(x=0, color='k')
plt.show()

def stability_region(scheme,x,y):
    N=len(x)
    M=len(y)
    rho=np.zeros([N,N])
    t,dt=0,1
    for i in range(N):
        for j in range(M):
            w=complex(x[i],y[j])
            U1=complex(1,0)
            r=scheme(U1,t,dt, lambda U,t : w*U)
            rho[i,j]=np.abs(r)
    return rho

def test_stability_region():
    x=np.linspace(-2,2,100)
    y=np.linspace(-2,2,100)
    rho=stability_region(euler,x,y)
    plt.contour(x,y,rho,np.linspace(0,1.,11))
    plt.axis('scaled')
    plt.show()