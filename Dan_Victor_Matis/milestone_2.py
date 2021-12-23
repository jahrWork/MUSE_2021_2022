# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 13:36:56 2021

@author: Dan, Matis, Victor
"""

from Integrate_Cauchy_problem import integrate_cauchy_problem
from Schemes import Euler, Euler_inverse, Crank_Nicholson
import numpy as np
import matplotlib. pyplot as plt

# Initial condition Kepler problem
U_0 = np.array([1., 
                0., 
                0.,
                1.])

# Function associated with the Kepler problem
def F(t, U) :
    
    module = (U[0]**2 + U[1]**2)**(3/2)
    return np.array([U[2], U[3], -U[0]/module, -U[1]/module])

n = 1000

delta_t = 0.01

Time = np.linspace(0, n*delta_t, n+1)
solution_euler = integrate_cauchy_problem(U_0, F, delta_t, n, Euler)
solution_inverse_euler = integrate_cauchy_problem(U_0, F, delta_t, n, Euler_inverse)
solution_CK = integrate_cauchy_problem(U_0, F, delta_t, n, Crank_Nicholson)


##Curves

##Euler method
plt.figure()

plt.plot(Time,solution_euler[0],'k.',label='x')
plt.plot(Time,solution_euler[1], 'r.',label='y')

plt.xlabel('Time')
plt.ylabel('Distance')     
plt.title('Euler methode, distance as a function of Time')  
plt.legend()
plt.grid()
      
plt.show()

plt.figure()

plt.plot(solution_euler[0],solution_euler[1],'k.',label='orbit')

plt.xlabel('x')
plt.ylabel('y')     
plt.title('Orbit Euler method')  
plt.legend()
plt.grid()
      
plt.show()

##Inverse Euler method
plt.figure()

plt.plot(Time,solution_inverse_euler[0],'k.',label='x')
plt.plot(Time,solution_inverse_euler[1], 'r.',label='y')

plt.xlabel('Time')
plt.ylabel('Distance')     
plt.title('Inverse Euler methode, distance as a function of Time')  
plt.legend()
plt.grid()
      
plt.show()

plt.figure()

plt.plot(solution_inverse_euler[0],solution_inverse_euler[1],'k.',label='orbit')

plt.xlabel('x')
plt.ylabel('y')     
plt.title('Orbit Inverse Euler method')  
plt.legend()
plt.grid()
      
plt.show()

##Crank Nicholson method
plt.figure()

plt.plot(Time,solution_CK[0],'k.',label='x')
plt.plot(Time,solution_CK[1], 'r.',label='y')

plt.xlabel('Time')
plt.ylabel('Distance')     
plt.title('Crank Nicholson methode, distance as a function of Time')  
plt.legend()
plt.grid()
      
plt.show()

plt.figure()

plt.plot(solution_CK[0],solution_CK[1],'k.',label='orbit')

plt.xlabel('x')
plt.ylabel('y')     
plt.title('Orbit Crank Nicholson method')  
plt.legend()
plt.grid()
      
plt.show()