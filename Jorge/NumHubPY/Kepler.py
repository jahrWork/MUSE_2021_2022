import numpy as np
import matplotlib.pyplot as plt
from Cauchy import *

def f(u,t):
    FF = np.concatenate( [u[3:], -u[0:3]/(np.linalg.norm(u[0:3], 2)**3)] )
    return FF

N, nn, x0, y0, z0, vx0, vy0, vz0 = input('Number of orbits = '), int(input('Number of points = ')), input('x0 = '), input('y0 = '), input('z0 = '), input('vx0 = '), input('vy0 = '), input('vz0 = ')
tmax, u0 = 2*np.pi*float(N), [x0, y0, z0, vx0, vy0, vz0]
print('Time = ', tmax)
print('h = ', tmax/nn)
sch=input('\nea, ei, cn, hn, rk4, am, lf, amx\nScheme: ')

[th,u] = ca(nn, tmax, u0, f, eval(sch))

plt.figure(0)
plt.plot(th, u[:,0:3])
plt.xlabel('Time [s]')
plt.ylabel('Distance [km]')
plt.legend(['x', 'y', 'z'])
plt.figure(1)
plt.plot(th, u[:,3:])
plt.xlabel('Time [s]')
plt.ylabel('Speed [km/s]')
plt.legend(['x', 'y', 'z'])
fig=plt.figure(2)
ax = plt.axes(projection='3d')
ax.plot3D(u[:,0], u[:,1], u[:,2])
plt.xlabel('x position')
plt.ylabel('y position')
ax.set_zlabel('z position')
ax.set_xlim3d(-1, 1)
ax.set_ylim3d(-1, 1)
ax.set_zlim3d(-1, 1)
plt.show()