import numpy as np
import matplotlib.pyplot as plt 

def DF(f, a, b, h, c):

    x = np.arange(a, b+h, h)
    n = 0
    fd = np.zeros(int((b - a)/h+1))
    for s in x:
        fd[n] = ( c[0]*f(s+2*h) + c[1]*f(s+h) + c[2]*f(s) + c[3]*f(s-h) + c[4]*f(s-2*h) )/h
        n = n + 1

    return x, fd

#def ftest(x):
#    return(x**4)
#c=[0, 1/2, 0, -1/2, 0]
#a=-2
#b=2
#h=0.1
#[x,fd]=DF(ftest,a,b,h,c)
#plt.plot(x,fd)
#plt.show()

def Jac(f, x, dx=1e-8):
    n = len(x)
    func = f(x)
    jac = np.zeros((n, n))
    for j in range(n):
        Dxj = (abs(x[j])*dx if x[j] != 0 else dx)
        x_plus = [(xi if k != j else xi + Dxj) for k, xi in enumerate(x)]
        jac[:, j] = (f(x_plus) - func)/Dxj
    return jac

def Jc(F, U0, eps=1e-8):
    N = len(U0)
    A = np.zeros([N,N])
    delta = np.zeros(N)
    t = 0
    for j in range(N):
            delta[:] = 0
            delta[j] = eps
            A[:,j] = ( F(U0+delta, 0.) - F(U0-delta, 0.) )/(2*eps)
    return A