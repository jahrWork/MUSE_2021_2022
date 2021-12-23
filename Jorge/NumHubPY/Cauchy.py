from ODE import *
import numpy as np

def ca(nn, tmax, u0, f, sch):
    h = tmax/(nn-1)
    th, u, u[0]=np.linspace(0, tmax, nn), np.zeros((int(nn), np.size(u0))), u0

    if sch == am or sch == lf:
        sch2 = 'rk4'
        a = 2
        for n in range(a):
            u[n+1,:] = eval(sch2)(h, u[n,:], th[n], f)
        for n in range(a,nn-1):
            u[n+1,:] = sch(h, u[n,:], th[n], f, u[n-1,:], u[n-2,:])
    
    elif sch == amx:
        sch2 = 'rk4'
        a = 4
        for n in range(a):
            u[n+1,:] = eval(sch2)(h, u[n,:], th[n], f)
        #us=[u[n-1,:],u[n-2,:],u[n-3,:],u[n-4,:]] #por algun motivo esto como input no vale
        for n in range(a, nn-1):
            u[n+1,:] = sch(h, u[n,:], th[n], f, [ u[n-1,:], u[n-2,:], u[n-3,:], u[n-4,:] ])
    
    else:
        for n in range(nn-1):
            u[n+1,:] = sch(h, u[n,:], th[n], f)
    return th, u