# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 12:23:20 2021

@author: Dan, Matis, Victor
"""

import numpy as np
from numpy.polynomial import Polynomial
from Integrate_Cauchy_problem import integrate_cauchy_problem
from Schemes import *
import matplotlib. pyplot as plt

## Absolute stability region of the oscillator problem ##

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


## Numerical resolution with different schemes
solution_Euler = integrate_cauchy_problem(X_0, Oscillator, delta_t, N, Euler)
solution_Euler_inverse = integrate_cauchy_problem(X_0, Oscillator, delta_t, N, Euler_inverse)
solution_Crank_Nicholson = integrate_cauchy_problem(X_0, Oscillator, delta_t, N, Crank_Nicholson)
solution_Runge_Kutta = integrate_cauchy_problem(X_0, Oscillator, delta_t, N, Runge_Kutta)
solution_Leap_Frog = integrate_cauchy_problem(X_0, Oscillator, delta_t, N, Leap_Frog)

##Curves
Time = np.linspace(0, N*delta_t, N+1)
plt.plot(Time, np.cos(Time), 'r', label='exact solution')
plt.plot(Time, solution_Leap_Frog[0], label='Leap Frog')
plt.plot(Time, solution_Euler[0], 'g', label='Euler')
plt.plot(Time, solution_Euler_inverse[0], 'y', label='Inverse Euler')
plt.plot(Time, solution_Crank_Nicholson[0], label='Crank Nicholson')
plt.plot(Time, solution_Runge_Kutta[0], 'k', label='Runge Kutta')
plt.legend()
plt.show()

## region of absolute stability

##definition of the different caracteristic polynom for each scheme
def polinomio_caracteristico_euler(omega) :
    return Polynomial([-1 - omega, 1])
 
def polinomio_caracteristico_inverse_euler(omega) :
    
    return Polynomial([-1, 1 - omega])

def polinomio_caracteristico_Crank_Nicholson(omega) :
    
    return Polynomial([-1 - 0.5*omega, 1 - 0.5*omega])

def polinomio_caracteristico_Leap_Frog(omega) :
    
    return Polynomial([-1, -2*omega, 1])

def polinomio_caracteristico_RK(omega) :
    
    return Polynomial([-1 - omega - (5/12)*omega**2 - (1/8)*omega**3 - (1/24)*omega**4, 1])


def is_in_stable_area(polinomio_caracteristico, w) :
    racines = (polinomio_caracteristico(w)).roots()
    if racines.shape[0] == 0 :
        return False
    for i in range(racines.shape[0]) :
        if np.abs(racines[i]) > 1 :
            return False
    return True
    
def region_absolute_stability(polinomio_caracteristico, minimum, maximum, step) :
    number_of_points = int((maximum - minimum)/step) + 1
    range_cpx = np.linspace(minimum, maximum, number_of_points)
    array_stability_area = []
    
    for part_real in range_cpx :
        for part_imaginary in range_cpx :
            w = part_real + part_imaginary*1j
            if is_in_stable_area(polinomio_caracteristico, w) :
                array_stability_area.append(w)
    
    tab_stability_area = np.array(array_stability_area, dtype = complex)
    
    return tab_stability_area

##Trazo de las curvas

tab_stability_area_euler = region_absolute_stability(polinomio_caracteristico_euler, -5, 5, 0.05)
part_real_euler = np.real(tab_stability_area_euler)
part_imag_euler = np.imag(tab_stability_area_euler)

tab_stability_area_inverse_euler = region_absolute_stability(polinomio_caracteristico_inverse_euler, -5, 5, 0.05)
part_real_inverse_euler = np.real(tab_stability_area_inverse_euler)
part_imag_inverse_euler = np.imag(tab_stability_area_inverse_euler)

tab_stability_area_CK= region_absolute_stability(polinomio_caracteristico_Crank_Nicholson, -5, 5, 0.05)
part_real_CK = np.real(tab_stability_area_CK)
part_imag_CK = np.imag(tab_stability_area_CK)

tab_stability_area_LF = region_absolute_stability(polinomio_caracteristico_Leap_Frog, -5, 5, 0.05)
part_real_LF = np.real(tab_stability_area_LF)
part_imag_LF = np.imag(tab_stability_area_LF)

tab_stability_area_RK = region_absolute_stability(polinomio_caracteristico_RK, -5, 5, 0.05)
part_real_RK = np.real(tab_stability_area_RK)
part_imag_RK = np.imag(tab_stability_area_RK)

plt.figure()
plt.scatter(part_real_euler, part_imag_euler)
plt.title('Absolute stability area Euler')
plt.xlabel('Part Real')
plt.ylabel('Part Imaginary')
plt.show()

plt.figure()
plt.scatter(part_real_inverse_euler, part_imag_inverse_euler)
plt.title('Absolute stability area Euler Inverse')
plt.xlabel('Part Real')
plt.ylabel('Part Imaginary')
plt.show()

plt.figure()
plt.scatter(part_real_CK, part_imag_CK)
plt.title('Absolute stability area Crank Nicholson')
plt.xlabel('Part Real')
plt.ylabel('Part Imaginary')
plt.show()

plt.figure()
plt.scatter(part_real_RK, part_imag_RK)
plt.title('Absolute stability area Runge Kutta')
plt.xlabel('Part Real')
plt.ylabel('Part Imaginary')
plt.show()

plt.figure()
plt.scatter(part_real_LF, part_imag_LF)
plt.title('Absolute stability area Leap Frog')
plt.xlabel('Part Real')
plt.ylabel('Part Imaginary')
plt.show()

