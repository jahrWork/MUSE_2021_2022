import numpy as np
from numerical_schemes import Euler_forward, Euler_backward, RungeKutta4, CrankNicholson

## Cauchy problem

# Solucion analitica (te da y1,y2 en un vector columna en funcion de t)

def Cauchy_analitic_solution(t):
    y_analitica = np.zeros(2)
    y_analitica[0] = t*np.exp(t**2.)
    y_analitica[1] = np.exp(t**2.)*(1 + 2.*t + (t**2.)/2)
    return y_analitica

# Funcion de Cauchy (te da dy en funcion de y,t)

def Cauchy_function(y,t):                           #y tiene que ser un vector columna de 2 componentes

    A = np.array([[2.*t, 0.],[1., 2.*t]])
    b = np.array([[np.exp(t**2.)],[2.*np.exp(t)]])
    homogeneous = 0
    F_Cauchy = np.zeros(2)
    for i in range(2):
        for j in range(2):
            homogeneous = A[i,j]*y[j] + homogeneous
            F_Cauchy[i]= homogeneous + b[i]
    return F_Cauchy

# Problema de Cauchy (te da la matriz (y1|y2) de las dos soluciones del problema de Cauchy)

def Cauchy_problem(y10,y20,t0,tf,N,scheme):

    ## Initial Conditions (y10,y20)

    U0 = np.array([y10,y20])

    ## Resolution

    Time = np.linspace(t0,tf,N+1) # vector fila

    Un = U0

    y = U0

    if scheme == "Euler_forward":
        for i in range(N):
            U_n1 = Euler_forward(Un,Time[i],Time[i+1],Cauchy_function)
            Un = U_n1
            y = np.vstack((y,U_n1))
    elif scheme == "Euler_backward":
        for i in range(N):
            U_n1 = Euler_backward(Un,Time[i],Time[i+1],Cauchy_function)
            Un = U_n1
            y = np.vstack((y,U_n1))
    elif scheme == "RungeKutta4":
        for i in range(N):
            U_n1 = RungeKutta4(Un,Time[i],Time[i+1],Cauchy_function)
            Un = U_n1
            y = np.vstack((y,U_n1))
    elif scheme == "CrankNicholson":
        for i in range(N):
            U_n1 = CrankNicholson(Un,Time[i],Time[i+1],Cauchy_function)
            Un = U_n1
            y = np.vstack((y,U_n1))
    else:
        raise Exception("mal puesto el esquema")

    return y            # la solucion esta por columnas: en la columna 0 la de y1 y en la columna 1 la de y2