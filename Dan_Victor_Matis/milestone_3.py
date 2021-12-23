# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 14:11:08 2021

@author: Dan, Matis, Victor
"""

import numpy as np
from Integrate_Cauchy_problem import integrate_cauchy_problem
from integration_error import integration_error, rate_convergence
from Schemes import *
import matplotlib. pyplot as plt

##Computation of the errors with the Oscillator Problem
## Initial condition

X_0 = np.array([1., 
                0.])

# Function associated with the oscillator problem

def Oscillator(t, X) :
    
    return np.array([X[1], 
                     -X[0]])

## Paramters for the numerical resolution

delta_t = 0.01
N = 1000

##Solutions of the oscillator problem with the four methods
solution_Euler = integrate_cauchy_problem(X_0, Oscillator, delta_t, N, Euler)
solution_Euler_inverse = integrate_cauchy_problem(X_0, Oscillator, delta_t, N, Euler_inverse)
solution_Crank_Nicholson = integrate_cauchy_problem(X_0, Oscillator, delta_t, N, Crank_Nicholson)
solution_Runge_Kutta = integrate_cauchy_problem(X_0, Oscillator, delta_t, N, Runge_Kutta)

##Errors according to the Richardson interpolation 
error_euler = integration_error(X_0, Oscillator, delta_t, N, Euler)
error_inverse_euler = integration_error(X_0, Oscillator, delta_t, N, Euler_inverse)
error_CK = integration_error(X_0, Oscillator, delta_t, N, Crank_Nicholson)
error_RK = integration_error(X_0, Oscillator, delta_t, N, Runge_Kutta)

#Rate Convergence plot data
log_N_euler, log_E_euler = rate_convergence(X_0, Oscillator, delta_t, 7, Euler)
log_N_inverse_euler, log_E_inverse_euler = rate_convergence(X_0, Oscillator, delta_t, 7, Euler_inverse)
log_N_CK, log_E_CK = rate_convergence(X_0, Oscillator, delta_t, 7, Crank_Nicholson)
log_N_RK, log_E_RK = rate_convergence(X_0, Oscillator, delta_t, 7, Runge_Kutta)

#Curves

Time = np.linspace(0, N*delta_t, N+1)





plt.figure()
plt.plot(Time, np.cos(Time), 'r', label = 'Exact solution')
plt.plot(Time, solution_Euler[0], 'g', label = 'Euler')
plt.xlabel('Time')
plt.ylabel('Solution')     
plt.title('Euler methode')  
plt.legend()
plt.show()

plt.figure()
plt.plot(Time, error_euler[0], '+')
plt.title('Euler error')
plt.xlabel('Time')
plt.ylabel('Error')  
plt.legend()
plt.show()

plt.figure()
plt.plot(Time, np.cos(Time), 'r', label = 'Exact solution')
plt.plot(Time, solution_Euler_inverse[0], 'g', label = 'Inverse Euler')
plt.xlabel('Time')
plt.ylabel('Solution')     
plt.title('Inverse Euler methode')  
plt.legend()
plt.show()

plt.figure()
plt.plot(Time, error_inverse_euler[0], '+')
plt.title('Inverse Euler error')
plt.xlabel('Time')
plt.ylabel('Error')  
plt.legend()
plt.show()

plt.figure()
plt.plot(Time, np.cos(Time), 'r', label = 'Exact solution')
plt.plot(Time, solution_Crank_Nicholson[0], 'g', label = 'Crank Nicholson')
plt.xlabel('Time')
plt.ylabel('Solution')     
plt.title('Crank Nicholson methode')  
plt.legend()
plt.show()

plt.figure()
plt.plot(Time, error_CK[0], '+')
plt.title('Crank Nicholson error')
plt.xlabel('Time')
plt.ylabel('Error')  
plt.legend()
plt.show()

plt.figure()
plt.plot(Time, np.cos(Time), 'r', label = 'Exact solution')
plt.plot(Time, solution_Runge_Kutta[0], 'g', label = 'Runge Kutta')
plt.xlabel('Time')
plt.ylabel('Solution')     
plt.title('Runge Kutta methode')  
plt.legend()
plt.show()

plt.figure()
plt.plot(Time, error_RK[0], '+')
plt.title('Runge Kutta error')
plt.xlabel('Time')
plt.ylabel('Error')  
plt.legend()
plt.show()

plt.figure() #RESULTADO NO CORRECTO
plt.plot(log_N_euler, (log_E_euler), 'r.',label='Euler')
plt.plot(log_N_inverse_euler,(log_E_inverse_euler),'k.',label='Inverse Euler')
plt.plot(log_N_CK, (log_E_CK), 'g.', label='Crank Nicholson')
plt.plot(log_N_RK, (log_E_RK), 'b.', label='Runge Kutta')
plt.title('Rate convergence')
plt.xlabel('log N')
plt.ylabel('|log E|')  
plt.grid()
plt.legend()
plt.show()