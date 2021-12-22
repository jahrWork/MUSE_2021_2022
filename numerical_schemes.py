import scipy as sp
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt

## euler explicito

def Euler_forward(un,tn,tn1,function):
    delta_t = tn1-tn
    un1 = un + delta_t*function(un,tn)
    return un1    # vector fila


## euler implicito

def Euler_backward(un,tn,tn1,function):
    delta_t = tn1-tn
    implicit_function = lambda un1, un, delta_t, tn1, function: un1 - un - delta_t*function(un1,tn1)
    un1 = optimize.newton(implicit_function,un,args=(un,delta_t,tn1,function),maxiter=10000)
    return un1
    

## runge kutta 4 orden

def RungeKutta4(un,tn,tn1,function):
    delta_t = tn1-tn
    k1 = function(un,tn)
    k2 = function(un + delta_t/2*k1, tn + 0.5*delta_t)
    k3 = function(un + delta_t/2*k2, tn + 0.5*delta_t)
    k4 = function(un + delta_t*k3, tn1)
    un1 = un + 1/6*delta_t*(k1 + 2*k2 + 2*k3 + k4)
    return un1


## Crank-Nicholson 

def CrankNicholson(un,tn,tn1,function):
    delta_t = tn1-tn
    implicit_function = lambda un1, un, tn, tn1, delta_t, function: (un1 - un)/delta_t - 0.5*(function(un1,tn1)+function(un,tn))
    un1 = optimize.newton(implicit_function,un,args=(un, tn, tn1, delta_t,function),maxiter=10000)
    return un1


## Leap-Frog

def LeapFrog(un,un_1,tn,tn1,function):
    delta_t = tn1 - tn
    un1 = un_1 + 2*delta_t*function(un,tn)
    return un1


## Regiones de estabilidad

def Stability_region_Euler_forward(): 

    x = np.linspace(-3,3,100)
    y = np.linspace(-3,3,100)

    M = len(x)
    N = len(y)

    rho = np.zeros([M,N])
    t, dt = 0, 1. 
    for i in range(M):
        for j in range(N):
            w = complex(x[i],y[j])
            r = 1 + w
            rho[j,i] = np.abs(r) 
    plt.contour(x,y,rho,np.linspace(0.,1.,11))
    plt.title('Stability region in w plane for Euler Forward method')
    plt.xlabel('Re(w)')
    plt.ylabel('Im(w)')
    plt.show()
    return rho

def Stability_region_Euler_backward(): 

    x = np.linspace(-3,3,100)
    y = np.linspace(-3,3,100)

    M = len(x)
    N = len(y)

    rho = np.zeros([M,N])
    t, dt = 0, 1. 
    for i in range(M):
        for j in range(N):
            w = complex(x[i],y[j])
            r = 1/(1-w)
            rho[j,i] = np.abs(r) 
    plt.contour(x,y,rho,np.linspace(0.,1.,11))
    plt.title('Stability region in w plane for Euler Backward method')
    plt.xlabel('Re(w)')
    plt.ylabel('Im(w)')
    plt.show()
    return rho

def Stability_region_CrankNicholson(): 

    x = np.linspace(-3,3,100)
    y = np.linspace(-3,3,100)

    M = len(x)
    N = len(y)

    rho = np.zeros([M,N])
    t, dt = 0, 1. 
    for i in range(M):
        for j in range(N):
            w = complex(x[i],y[j])
            r = (1 + w/2)/(1-w/2)
            rho[j,i] = np.abs(r) 
    plt.contour(x,y,rho,np.linspace(0.,1.,11))
    plt.title('Stability region in w plane for Crank Nicolson method')
    plt.xlabel('Re(w)')
    plt.ylabel('Im(w)')
    plt.show()
    return rho

def Stability_region_RungeKutta4(): 

    x = np.linspace(-3,3,100)
    y = np.linspace(-3,3,100)

    M = len(x)
    N = len(y)

    rho = np.zeros([M,N])
    t, dt = 0, 1. 
    for i in range(M):
        for j in range(N):
            w = complex(x[i],y[j])
            #r = 1 + w + w**2/2
            r = -1 - w - w**2/2 - w**3/6-w**4/24
            rho[j,i] = np.abs(r) 
    plt.contour(x,y,rho,np.linspace(0.,1.,11))
    plt.title('Stability region in w plane for Runge Kutta 4 method')
    plt.xlabel('Re(w)')
    plt.ylabel('Im(w)')
    plt.show()
    return rho



def Stability_region_LeapFrog(): 

    x = np.linspace(-3,3,100)
    y = np.linspace(-3,3,100)

    M = len(x)
    N = len(y)

    rho = np.zeros([M,N])

    t, dt = 0, 1. 
    w = np.zeros(360)
    for theta in range(0,360):
            w[theta] = np.sin(theta*2*np.pi/360)
    plt.title('Stability region in w plane for Leap-Frog method')
    plt.xlabel('Re(w)')
    plt.ylabel('Im(w)')
    plt.plot(np.zeros(360),w)
    plt.show()

