import numpy as np
import matplotlib.pyplot as plt
from Cauchy import *

def f_N(u,t):
    return f_Np(u, Nb, Nc)

#N, nn = input('Number of orbits = '), int(input('Number of points = '))
#tmax, u0 = 2*np.pi*float(N), [x1, y1, z1, vx1, vy1, vz1, x2, y2, z2, vx2, vy2, vz2]
#print('Time = ', tmax)
#print('h = ', tmax/nn)
sch=input('\nea, ei, cn, hn, rk4, am, lf, amx\nScheme: ')

N = int(input('Number of points = '))-1
Nb = 4 #int(input('Number of bodies = '))
Nc = 3 #int(input('Number of coordinates = '))
nn = N+1
tmax = 4*3.14 #input('Tmax = ')

def Starting_conditions(Nc, Nb):
    u0 = np.zeros(2*Nc*Nb)
    u1 = np.reshape(u0, (Nb, Nc, 2))
    r0 = np.reshape(u1[:,:,0], (Nb, Nc))
    v0 = np.reshape(u1[:,:,1], (Nb, Nc))

    r0[0,:] = [1, 0, 0]
    v0[0,:] = [0, 0.4, 0]

    r0[1,:] = [-1, 0, 0]
    v0[1,:] = [0, -0.4, 0]

    r0[2,:] = [0, 1, 0]
    v0[2,:] = [-0.4, 0, 0]

    r0[3,:] = [0, -1, 0]
    v0[3,:] = [0.4, 0, 0]

    return u0

def f_Np(u, Nb, Nc):

    us = np.reshape(u, (Nb, Nc, 2))
    F = np.zeros(len(u))
    dus = np.reshape(F, (Nb, Nc, 2))

    r = np.reshape(us[:, :, 0], (Nb, Nc))
    v = np.reshape(us[:, :, 1], (Nb, Nc))

    drdt = np.reshape(dus[:, :, 0], (Nb, Nc))
    dvdt = np.reshape(dus[:, :, 1], (Nb, Nc))

    dvdt[:,:] = 0

    for i in range(Nb):
        drdt[i,:] = v[i,:]
        for j in range(Nb):
            if j != i:
                d = r[j,:] - r[i,:]
                dvdt[i,:] = dvdt[i,:] + d[:]/np.linalg.norm(d)**3

    return F

u0 = Starting_conditions(Nc, Nb)
[th,u] = ca(nn, tmax, u0, f_N, eval(sch))

us = np.reshape(u, (N+1, Nb, Nc, 2))
r = np.reshape(us[:,:,:,0], (N+1, Nb, Nc))

ax = plt.axes(projection='3d')
for i in range(Nb):
    ax.plot3D(r[:, i, 0], r[:, i, 1], r[:, i, 2])

plt.xlabel('x position')
plt.ylabel('y position')
ax.set_zlabel('z position')
lim=1
ax.set_xlim3d(-lim, lim)
ax.set_ylim3d(-lim, lim)
ax.set_zlim3d(-lim, lim)
plt.show()