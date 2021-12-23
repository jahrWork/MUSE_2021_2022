import numpy as np
from a_zeros import *

def ea(h, u, th, f):   
    return u + f(u, th)*h

def ei(h, u, th, f):
    u1 = u
    def F(y, t):
        return(y - u - h*f(y, th + h))
    u1 = new2(u1,F)
    return u1

def cn(h, u, th, f):
    u1 = u
    def F(y, t):
        return(y - u - h/2*(f(y, th + h) + f(u, th)))
    u1 = new2(u1, F)
    return u1

def hn(h, u, th, f):
    uu = u + f(u, th)*h
    return u + h/2*(f(u, th) + f(uu, th))

def rk4(h, u, th, f):
    k = np.zeros((4, np.size(u)))
    k[0] = f(u, th)
    k[1] = f(u + k[0]*h/2, th + h/2)
    k[2] = f(u + k[1]*h/2, th + h/2)
    k[3] = f(u + k[2]*h, th + h)
    return u + h/6*([1,2,2,1]@k)

def am(h, u, th, f, us1, us2):
    uu = u + f(u, th)*h*23/12 - f(us1, th - h)*h*4/3 + f(us2, th - 2*h)*h*5/12
    return u + f(u, th)*h*19/24 - f(us1, th - h)*h*5/24 + f(us2, th - 2*h)*h*1/24 + f(uu, th + h)*h*3/8

def amx(h, u, th, f, us):
    uu = u + f(u, th)*h*1901/720 - f(us[0], th - h)*h*1387/360 + f(us[1], th - 2*h)*h*109/30 - f(us[2], th - 3*h)*h*637/360 + f(us[3], th - 4*h)*h*251/720
    #uu=u+h*[1901/720, 1387/360, 109/30, 637/360, 251/720]@[f(u,th), f(us[0],th-h), f(us[1],th-2*h), f(us[2],th-3*h), f(us[3],th-4*h)]
    return u + f(u, th)*h*646/720 - f(us[0], th - h)*h*264/720 + f(us[1], th - 2*h)*h*106/720 - f(us[2], th - 3*h)*h*19/720 + f(uu, th + h)*h*251/720

def lf(h, u, th, f, us1, us2):
    return us1 + f(u, th)*h*2


#def dF(y):
#    return(1-h*df_y)
#u1=new(u1,F,dF)