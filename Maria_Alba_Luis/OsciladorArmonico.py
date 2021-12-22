import numpy as np
from numerical_schemes import Euler_forward, Euler_backward, RungeKutta4, CrankNicholson, LeapFrog


## Oscilador armonico

# Funcion del oscilador (te da dU en funcion de U)

def LinearOscillator(U,t):
    x = U[0:2]
    A=np.array([[0 , 1],[-1, 0]])
    F_LinearOscillator=np.zeros(2)
    for i in range(2):
        homogeneous=0
        for j in range(2):
            F_LinearOscillator[i] = A[i,j]*x[j] + F_LinearOscillator[i]
    return F_LinearOscillator

#Movimiento oscilador (te da la matriz (x|xprima|) de posicion y velocidad en los dos ejes

def Oscillator(x0,t0,tf,N,scheme):

    ## Initial Conditions (position and velocity)

    U0 = np.array([x0[0], x0[1]]) # vector fila

    ## Orbit_resolution

    Time = np.linspace(t0,tf,N+1) # vector fila

    Un = U0

    oscilador_mov = U0

    if scheme == "Euler_forward":
        for i in range(N):
            U_n1 = Euler_forward(Un,Time[i],Time[i+1],LinearOscillator)
            Un = U_n1
            oscilador_mov = np.vstack((oscilador_mov,U_n1))
    elif scheme == "Euler_backward":
        for i in range(N):
            U_n1 = Euler_backward(Un,Time[i],Time[i+1],LinearOscillator)
            Un = U_n1
            oscilador_mov = np.vstack((oscilador_mov,U_n1))
    elif scheme == "RungeKutta4":
        for i in range(N):
            U_n1 = RungeKutta4(Un,Time[i],Time[i+1],LinearOscillator)
            Un = U_n1
            oscilador_mov = np.vstack((oscilador_mov,U_n1))
    elif scheme == "CrankNicholson":
        for i in range(N):
            U_n1 = CrankNicholson(Un,Time[i],Time[i+1],LinearOscillator)
            Un = U_n1
            oscilador_mov = np.vstack((oscilador_mov,U_n1))
    elif scheme == "LeapFrog":
        for i in range(N):
            if i==0:
                Un=Euler_forward(U0,Time[i],Time[i+1],LinearOscillator)
                U_n1 = LeapFrog(Un,U0,Time[i],Time[i+1],LinearOscillator)
                U0=Un
                Un=U_n1
                oscilador_mov = np.vstack((oscilador_mov,U_n1))
            else:
                U_n1 = LeapFrog(Un,U0,Time[i],Time[i+1],LinearOscillator)
                U0=Un
                Un=U_n1
                oscilador_mov = np.vstack((oscilador_mov,U_n1))
    else:
        raise Exception("mal puesto el esquema")

    return oscilador_mov, Time
