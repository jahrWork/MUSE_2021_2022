import numpy as np
from numerical_schemes import Euler_forward, Euler_backward, RungeKutta4, CrankNicholson


## Kepler orbits 

# Funcion de Kepler (te da dU en funcion de U)

def Kepler_function(U,t):
    r = U[0:2]
    drdt = U[2:4]
    F_Kepler = np.append(drdt,-r/np.linalg.norm(r)**3, axis = 0)   # vector fila
    return F_Kepler

# Órbita de Kepler (te da la matriz (x|y|xprima|yprima) de posicion y velocidad en los dos ejes)

def Kepler_orbit(x0,y0,xdot0,ydot0,t0,tf,N,scheme):

    ## Initial Conditions (position and velocity)

    U0 = np.array([x0,y0,xdot0,ydot0]) # vector fila

    ## Orbit_resolution

    Time = np.linspace(t0,tf,N+1) # vector fila

    Un = U0

    orbitas = U0

    if scheme == "Euler_forward":
        for i in range(N):
            U_n1 = Euler_forward(Un,Time[i],Time[i+1],Kepler_function)
            Un = U_n1
            orbitas = np.vstack((orbitas,U_n1))
    elif scheme == "Euler_backward":
        for i in range(N):
            U_n1 = Euler_backward(Un,Time[i],Time[i+1],Kepler_function)
            Un = U_n1
            orbitas = np.vstack((orbitas,U_n1))
    elif scheme == "RungeKutta4":
        for i in range(N):
            U_n1 = RungeKutta4(Un,Time[i],Time[i+1],Kepler_function)
            Un = U_n1
            orbitas = np.vstack((orbitas,U_n1))
    elif scheme == "CrankNicholson":
        for i in range(N):
            U_n1 = CrankNicholson(Un,Time[i],Time[i+1],Kepler_function)
            Un = U_n1
            orbitas = np.vstack((orbitas,U_n1))
    else:
        raise Exception("mal puesto el esquema")

    return orbitas, Time


