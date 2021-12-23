import numpy as np
import matplotlib.pyplot as plt
from Cauchy import *

def f(u, t):
    FF = np.array([[0,1], [-1,0]])@u
    return(FF)

N, nn, x0, dx0 = input('Number of periods = '), int(input('Number of points = ')),  input('x0 = '), input('dx0 = ')
tmax, y0 = 2*np.pi*float(N), [x0,dx0]
print('Time = ', tmax)
print('h = ', tmax/nn)
sch = input('\nea, ei, cn, hn, rk4, am, lf, amx\nScheme: ')

[th,u] = ca(nn, tmax, y0, f, eval(sch))

plt.figure(0)
plt.plot(th,u[:,0])
plt.xlabel('Time [s]')
plt.ylabel('Distance [km]')
plt.legend(['x', 'y', 'z'])
plt.figure(1)
plt.plot(th,u[:,1])
plt.xlabel('Time [s]')
plt.ylabel('Speed [km/s]')
plt.legend(['x', 'y', 'z'])
plt.show()