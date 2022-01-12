#   ------------------------------------------------------------------
#                           CAUCHY PROBLEM 
#                    AMPLIACIÓN DE MATEMÁTICAS 1
#               MÁSTER UNIVERSITARIO DE SISTEMAS ESPACIALES
#   ------------------------------------------------------------------




# Imports
from numpy import zeros
from numpy import pi
import matplotlib as plt
from temporal_schemes import Euler, RK4, CN


# Main Cauchy Problem 

# Force of the Kepler movement: F_Kepler
#def F_kep(U):
#    # here, U is a vector
#    r = U[0:2]
#    drdt = U[2:4]
#    F_kep = np.concatenate([drdt,- r/(np.linalg.norm(r)**3)])
#    return F_kep

def CauchyProblemSol(Time_Domain, U0, Temporal_Scheme, Differential_operator):

    ##########print(Temporal_Scheme)

    N_steps = len(Time_Domain) - 1

    Nv = len(U0)
    Solution = zeros( (N_steps + 1, Nv), dtype=float)
    Solution[0, :] = U0

    for n in range(0, N_steps):

        t1 = Time_Domain[n]
        t2 = Time_Domain[n+1]

        Solution[n+1,:] = Temporal_Scheme(t1, t2, Solution[n,:], Differential_operator)
        #if Temporal_Scheme == 1: Solution[i,:] = Euler(t1, t2, Solution[i-1,:], Differential_operator)
        #if Temporal_Scheme == 2: Solution[i,:] = temporal_schemes.InverseEuler(t1,t2,U[i-1,:],U[i,:])
        #if Temporal_Scheme == 3: Solution[i,:] = temporal_schemes.CrankNicholson(t1,t2,U[i-1,:],U[i,:])
        #if Temporal_Scheme == 4: Solution[i,:] = temporal_schemes.RK4(t1,t2,U[i-1,:],U[i,:])
    return Solution


