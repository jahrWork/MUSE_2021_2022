import numpy as np
import matplotlib.pyplot as plt
from Cauchy import *

def f(u, t):
    FF = np.concatenate([u[3:], -u[0:3]/(np.linalg.norm(u[0:3], 2)**3)])
    return FF

#N, nn, x0, y0, z0, vx0, vy0, vz0, df_y=input('Number of orbits = '), int(input('Starting number of points = ')), input('x0 = '), input('y0 = '), input('z0 = '), input('vx0 = '), input('vy0 = '), input('vz0 = '), np.array([0,0,0,0,0,0])
N, nn, x0, y0, z0, vx0, vy0, vz0 = 1, int(input('Starting number of points = ')), 1, 0, 0, 0, 1, 0
tmax = 2*np.pi*float(N)
print('Time = ', tmax)
print('h = ', tmax/(nn - 1))
u0 = [x0, y0, z0, vx0, vy0, vz0]
#sch=input('\nea, ei, cn, hn, rk4, am, lf, amx\nScheme: ')
sch_label = 'rk4'

def plottt(th, u):
    fig=plt.figure(0)
    ax = plt.axes(projection='3d')
    ax.plot3D(u[:,0], u[:,1], u[:,2])
    plt.xlabel('x position')
    plt.ylabel('y position')
    ax.set_zlabel('z position')
    #ax.set_xlim3d(-1, 1)
    #ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)
    plt.show()

def ErrorRE(n, nn, tmax, u0, f, sch):
    [th1,u1] = ca(nn*n, tmax, u0, f, sch)
    [th2,u2] = ca(nn*2*n-1, tmax, u0, f, sch)
    print()
    print(nn*n)
    print(u1[-1,:])
    print(nn*2*n-1)
    print(u2[-1,:])
    print('delta', np.linalg.norm(u1[-1,:] - u2[-1,:]))
    #plottt(th1,u1)
    #plottt(th2,u2)
    return np.linalg.norm(u1[-1,:] - u2[-1,:])

def estimate_error(q,n,t):
    N = np.arange(N+1)
    E = np.exp(q*np.log(t) - q*np.log(N))
    plot(log(N),log(E))

a = 10 #30,30,80
UU = np.zeros(a - 1)
NN = np.arange(1, a)*nn
#NN=nn*2**(np.arange(1,a)-1)

for n in range(1, a):
    print('\n', n)
    UU[n-1] = ErrorRE(n, nn, tmax, u0, f, eval(sch_label))

q = -np.polyfit(np.log(NN), np.log(UU), 1)
print(UU)
print(np.log(UU))
print('q ~= ', q[0])
plt.plot(np.log(NN), np.log(UU), '*')
plt.plot(np.log(NN), -q[0]*np.log(NN) - q[1])
plt.axis('scaled')
plt.show()