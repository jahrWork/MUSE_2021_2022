#   ------------------------------------------------------------------
#                           Milestone #3
#                    AMPLIACIÓN DE MATEMÁTICAS 1
#               MÁSTER UNIVERSITARIO DE SISTEMAS ESPACIALES
#   ------------------------------------------------------------------


# Imports
from numpy import array, linspace, pi
from kepler_orbit import F_kep
from cauchyproblem import CauchyProblemSol
from temporal_schemes import Euler, RK4, CN
from temporalerror import Extrapolacion_Richardson, Velocidad_Convergencia
import matplotlib.pyplot as plt
from numpy.linalg import norm, lstsq


def M3():

    ############################
    Temporal_Scheme = RK4 # Euler, RK4, CN (ojo con Euler que o subir mucho los puntos o bajar mucho el tiempo)
    U0 = array( [1, 0, 0, 1])
    N = 50
    number_points= 12
    Time = linspace( 0, 12*pi, N )
    C= 2
    order_Richardson = 4
    ############################

    #E = norm(Extrapolacion_Richardson(U0,Time, F_kep, Temporal_Scheme, C, order_Richardson))

    #plt.plot(E, Time, color ='b')    
    #plt.axis('square')
    #plt.xlabel("E", size = 12)
    #plt.ylabel("Time", size = 12)
    #plt.title("Richarson Error", size = 16)
    #plt.grid(True)
    #plt.show()



    log_N, log_E = Velocidad_Convergencia(U0, Time, F_kep, Temporal_Scheme, number_points) 
    
    plt.plot(log_N, log_E, color ='b')    
    plt.axis('square')
    plt.xlabel("log N", size = 12)
    plt.ylabel("log U2 - U1", size = 12)
    plt.title("Convergence Rate", size = 16)
    plt.grid(True)
    plt.show()

