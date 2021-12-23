import numpy as np

def pmed(a, b, n, g):
    h = (b - a)/n
    s = np.linspace(a+h/2, b-h/2, n)
    I = 0
    for i in np.arange(0, n, 1):
        I = I + g(s[i])*h
    return I

def nc(a, b, g, n, deg):
    deg = eval('nc'+str(deg))
    h = (b - a)/n
    s = np.linspace(a, b, n+1)
    I = 0
    for i in np.arange(0, n, 1):
        I = I + deg(g, s, i, h)
    return I

def nc1(g, s, i, h):
    return h*(g(s[i]) + g(s[i+1]))/2

def nc2(g, s, i, h):
    return h*( g(s[i]) + g(s[i+1]) + 4*g((s[i] + s[i+1])/2) )/6

def nc3(g, s, i, h):
    return h*( g(s[i]) + 3*g((2*s[i] + s[i+1])/3) + 3*g((s[i] + 2*s[i+1])/3) + g(s[i+1]) )/8

def nc4(g, s, i, h):
    return h*( 7*g(s[i]) + 32*g((3*s[i] + s[i+1])/4) + 12*g((s[i] + s[i+1])/2) + 32*g((s[i] + 3*s[i+1])/4) + 7*g(s[i+1]) )/90

def nc5(g, s, i, h):
    return h*( 19*g(s[i]) + 75*g((4*s[i] + s[i+1])/5) + 50*g((3*s[i] + s[i+1])/5) + 50*g((s[i] + 3*s[i+1])/5) + 75*g((s[i] + 4*s[i+1])/5) + g(s[i+1]) )/288

def nc6(g, s, i, h):
    return h*( 41*g(s[i]) + 216*g((5*s[i] + s[i+1])/6) + 27*g((4*s[i] + 2*s[i+1])/6) + 272*g((s[i] + s[i+1])/2) + 27*g((2*s[i] + 4*s[i+1])/6) + 216*g((s[i] + 5*s[i+1])/6) + 41*g(s[i+1]) )/840

#def g(x):
#    return x**4
#print(nc(0,5,g,1,6))

#def gaussleg(a,b,f):
#    y_ref = 0
#    alpha_ref = 2; 
#    y=(a+b)/2+(b-a)/2*y_ref
#    alpha=(b-a)/2*alpha_ref
#    return(np.sum(alpha*f(y)))