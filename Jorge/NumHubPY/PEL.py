import numpy as np
import matplotlib.pyplot as plt
from Cauchy import *
import scipy.io

def f(u,t):
    a=5.143e-5
    rho=1.802e3
    cs=1.527e3
    l1=0.254
    l2=0.4399
    n=0.3
    Ag=0.0508
    L=2.9210

    FF=a*(rho*a*cs*(l1+2*l2-4*u)*6*L/Ag)**(n/(1-n))
    return(FF)

nn, y0, df_y=int(input('Number of points = ')), input('y0 = '), np.array([0])
tmax, u0=25.7583, [y0]
print('Time = ',tmax)
print('h = ',tmax/nn)
sch=input('\nea, ei, cn, hn, rk4, am, lf, amx\nScheme: ')

[th,u]=ca(nn,tmax,u0,f,df_y,eval(sch))
print(u[-1])
plt.figure(0)
plt.plot(th,u[:,0])
plt.xlabel('Time [s]')
plt.ylabel('Distance [km]')
plt.legend(['x', 'y', 'z'])
plt.show()

a=5.143e-5
rho=1.802e3
cs=1.527e3
l1=0.254
l2=0.4399
n=0.3
Ag=0.0508
L=2.9210

P=(rho*a*cs*(l1+2*l2-4*u)*6*L/Ag)**(1/(1-n))
print(P)
scipy.io.savemat('test.mat', dict(P=P))