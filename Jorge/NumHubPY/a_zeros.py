import numpy as np
from a_DF import *

def bisec(a, b, f):
    x = (b + a)/2
    n = 0
    while n < 100:
        n = n + 1
        if f(x) == 0:
            x
        else:
            if f(x)*f(b) < 0:
                a = x
            elif f(x)*f(a) < 0:
                b = x
        x = (a + b)/2
    return x

def new(u1, F, dF):
    p = 0
    while np.linalg.norm(F(u1)) > 1e-15 and p < 20:
        p = p + 1
        u1 = u1 - F(u1)/dF(u1)
    return u1

def new2(u1, F):
    p = 0
    while np.linalg.norm(F(u1, 0.)) > 1e-15 and p < 100:
        p = p + 1
        u1 = u1 - np.linalg.lstsq(Jc(F, u1, 1e-8), F(u1, 0.))[0]
    return u1


#def fixpt(x,phi):
#    #def phi(x):
#    #    return(0.5*x+5)
#    n=0
#    while n<100:
#        n=n+1
#        x=phi(x)
#    print('Number of iterations:',n)
#    return(x)