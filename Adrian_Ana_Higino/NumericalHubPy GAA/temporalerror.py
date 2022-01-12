#   ------------------------------------------------------------------
#                           TEMPORAL ERRORS 
#                    AMPLIACIÓN DE MATEMÁTICAS 1
#               MÁSTER UNIVERSITARIO DE SISTEMAS ESPACIALES
#   ------------------------------------------------------------------

from cauchyproblem import CauchyProblemSol
from kepler_orbit import F_kep
from temporal_schemes import Euler, RK4, CN
from numpy import array, linspace, pi, zeros, log, vstack, ones, polyfit, transpose
from numpy.linalg import norm, lstsq


def error_U(U0, Time_Domain, Differential_operator, Temporal_Scheme, C):
         # U0 is the initial vector, U1 is the solved of the last iteration (save time)
    N_steps = len(Time_Domain)
    Error_array= zeros((N_steps-1, len(U0)))
    t1= Time_Domain                                                        # Evalúo el Euler, estimo orden 1
    t2 = linspace( Time_Domain[0], Time_Domain[-1], C*N_steps ) 

    U1 = CauchyProblemSol(t1, U0, Temporal_Scheme, Differential_operator)                # Resuelvo problema de Cauchy para la función U
    U2 = CauchyProblemSol(t2, U0, Temporal_Scheme, Differential_operator)                         # Resuelvo problema de Cauchy para la función V

    error_U= U2[-1,:] - U1[-1,:]

    return error_U


def estim_order(N, error_U):
    log_N= log(N) 
    y_aux= zeros((len(error_U), 1))
    log_EU= log(error_U)
    log_N = transpose(log_N)
    log_EU = transpose(log_EU)
    m= polyfit(log_N, log_EU, 1)
    q= abs(round(m[0]))
    print('order q=', q)

    return q


def filtrado(N, vector, val_min, val_max):
    j= 0
    vector_filt_aux= zeros(len(N)) # inicializarlo como array
    N_filt_aux= zeros(len(N))
    for i in range (0, len(N)-1):
        norm_vector= norm(vector[i,:])
        N_aux= N[i]
        if norm_vector<= val_max:
            if norm_vector>= val_min:
                vector_filt_aux[j]= norm_vector
                N_filt_aux[j]= N_aux
                j=j+1

    vector_filt= zeros((j))
    N_filt= zeros((j))
    for i in range(0, j):
        vector_filt[i] = vector_filt_aux[i]
        N_filt[i]= N_filt_aux[i]
    return N_filt, vector_filt


def error_order(C, N, error_U):
    N_filt, error_U_filter= filtrado(N, error_U, 1e-11, 1e-3) # el resultado es un vector ya con la norma hecha
    q= estim_order(N_filt, error_U_filter)
    error_order= (1-(1/C)**q)

    return error_order




def Extrapolacion_Richardson(U0, Time_Domain, Differential_operator, Temporal_Scheme, C, order):
        # U0 is the initial vector, U1 is the solved of the last iteration (save time)
    N_steps = len(Time_Domain)
    Error_array= zeros((N_steps-1, len(U0)))                                                          # Evalúo el Euler, estimo orden 1
    t2 = linspace( Time_Domain[1], Time_Domain[-1], C*N_steps ) 

    U1 = CauchyProblemSol(Time_Domain, U0, Temporal_Scheme, Differential_operator)                # Resuelvo problema de Cauchy para la función U
    U2 = CauchyProblemSol(t2, U0, Temporal_Scheme, Differential_operator)                         # Resuelvo problema de Cauchy para la función V
    for i in range (0, N_steps-1):
        Error_array[i,:] = ( U2[C*i,:] - U1[i,:] ) / ( 1 - (1/C)**order)      # Hago la extrapolación de Richardson para cada paso temporal

    return Error_array

    

def Velocidad_Convergencia(U0, Time_Domain, Differential_operator, Temporal_Scheme, n_points):
    C=2

    N= zeros(n_points)
    log_N= zeros(n_points)
    log_E_U= zeros((n_points))
    E_U= zeros((n_points, len(U0)))
    E_q= zeros((n_points, len(U0)))

    N= zeros(n_points)
    log_N= N
    log_E_U= N
    E_U= zeros((n_points, len(U0)))
    E_q= E_U

    t1= Time_Domain
    N_steps= len(t1)

    vector= zeros((n_points, 2))

    for i in range (0, n_points):
       
        N[i]= N_steps

        E_U[i,:]= error_U(U0, t1, Differential_operator, Temporal_Scheme, C)
        log_E_U[i]= log(norm(E_U[i,:]))
        t1= linspace( Time_Domain[1], Time_Domain[-1], C*N_steps)  
        N_steps = len(t1)
             
    E_q= error_order(C, N, E_U)   # es constante al ser C cte
    log_N= log(N)
    log_E_global= log_E_U - log(E_q)

    return log_N, log_E_global
