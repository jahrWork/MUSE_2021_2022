# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 11:24:19 2021

@author: Dan, Matis, Victor
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so
from Integrate_Cauchy_problem import integrate_cauchy_problem
from Schemes import Runge_Kutta

def distancia_1_3BP(x, y, mu) :
    return ((x + mu)**2 + y**2)**(1/2)

def distancia_2_3BP(x, y, mu) :
    return ((x + mu - 1)**2 + y**2)**(1/2)

def function_Lagrange(x, mu) :
    return x - (1-mu)*(x+mu)/(np.abs(x+mu))**3 - mu*(x+mu-1)/(np.abs(x-1+mu))**3

def function_Kepler_3B(t, R) :
    global mu
    
    distance_r1 = distancia_1_3BP(R[0], R[1], mu)
    distance_r2 = distancia_2_3BP(R[0], R[1], mu)
    
    return np.array([R[2], 
                     R[3], 
                     R[0] + 2*R[3] - (1-mu)*(R[0] + mu)*distance_r1**(-3) - mu*(R[0]+ mu - 1)*distance_r2**(-3), 
                     R[1] - 2*R[2] - (1-mu)*R[1]*distance_r1**(-3) - mu*R[1]*distance_r2**(-3)])

def d_U_x_x(x, y, mu) :
    distance_r1 = distancia_1_3BP(x, y, mu)
    distance_r2 = distancia_2_3BP(x, y, mu)
    
    return  -(1-mu)*distance_r1**(-3/2)+3*(1-mu)*(mu+x)**2*distance_r1**(-5/2)-mu*distance_r2**(-3/2)+3*mu*(x-1+mu)**2*distance_r2**(5/2) + 1

def d_U_x_y(x, y, mu) :
    distance_r1 = distancia_1_3BP(x, y, mu)
    distance_r2 = distancia_2_3BP(x, y, mu)
    
    return 3*y*((1-mu)*(mu+x)*distance_r1**(-5/2)+mu*(x-1+mu)*distance_r2**(-5/2))

def d_U_y_x(x, y, mu) :
    distance_r1 = distancia_1_3BP(x, y, mu)
    distance_r2 = distancia_2_3BP(x, y, mu)   
    
    return 3*y*((1-mu)*(mu+x)*distance_r1**(-5/2)+mu*(x-1+mu)*distance_r2**(-5/2))

def d_U_y_y(x, y, mu) :
    distance_r1 = distancia_1_3BP(x, y, mu)
    distance_r2 = distancia_2_3BP(x, y, mu)
    
    return -(1-mu)*distance_r1**(-3/2)+3*(1-mu)*(y**2)*distance_r1**(-5/2)-mu*distance_r2**(-3/2)+3*mu*(y**2)*distance_r2**(5/2)+1

def matrix_system_Lagrange(mu, x_Lagrange, y_Lagrange) :
    
    return np.array([[0, 0, 1, 0],
                     [0, 0, 0, 1], 
                     [d_U_x_x(x_Lagrange, y_Lagrange, mu), d_U_x_y(x_Lagrange, y_Lagrange, mu), 0, 2], 
                     [d_U_y_x(x_Lagrange, y_Lagrange, mu), d_U_y_y(x_Lagrange, y_Lagrange, mu), -2, 0]])

def function_around_Lagrange(t, R) :

    global mu
    global x_Lagrange
    global y_Lagrange
    
    return np.dot(matrix_system_Lagrange(mu, x_Lagrange, y_Lagrange), R)    

def Lagrange_points(mu) :
    array_Lagrange_points = []
    beta = (mu/3)**(1/3)
    gamma = beta*(1-beta/3 - (1/9)*beta**2)
    
    #Lagrange points on the axis y = 0
    pos_y = 0
    
    approximate_value_x1 = 1-mu+gamma + 0.1
    pos_x_L1 = so.newton(lambda x : function_Lagrange(x, mu), approximate_value_x1, x1 = approximate_value_x1 + 0.5, tol=10**(-9), maxiter=1000)
    array_Lagrange_points.append((pos_x_L1, pos_y))
    
    approximate_value_x2 = 1-mu-gamma - 0.1
    pos_x_L2 = so.newton(lambda x : function_Lagrange(x, mu), approximate_value_x2, x1 = approximate_value_x2 +0.5, tol=10**(-9), maxiter=1000)
    array_Lagrange_points.append((pos_x_L2, pos_y))
    
    approximate_value_x3 = -(1+(5/12)*mu) - 0.1
    pos_x_L3 = so.newton(lambda x : function_Lagrange(x, mu), approximate_value_x3, x1 = approximate_value_x3 + 0.5, tol=10**(-9), maxiter=1000)
    array_Lagrange_points.append((pos_x_L3, pos_y))
    
    #Lagrange points where r1 = r2
    pos_x_L4 = 0.5 - mu
    pos_x_L5 = 0.5 - mu
    pos_y_L4 = (3/4)**0.5
    pos_y_L5 = -(3/4)**0.5
    
    array_Lagrange_points.append((pos_x_L4, pos_y_L4))
    array_Lagrange_points.append((pos_x_L5, pos_y_L5))
    tab_Lagrange_points = np.zeros((5, 2))
    for i in range(5) :
        tab_Lagrange_points[i, 0] = array_Lagrange_points[i][0]
        tab_Lagrange_points[i, 1] = array_Lagrange_points[i][1]
        
    return tab_Lagrange_points

def stability_lagrange_points(mu, tab_Lagrange) :
    
    lista_eigen_values = []
    for i in range(5) :
        x_Lagrange = tab_Lagrange[i, 0]
        y_Lagrange = tab_Lagrange[i, 1]
        matrix_system = matrix_system_Lagrange(mu, x_Lagrange, y_Lagrange)
        
        eigen_values = np.linalg.eigvals(matrix_system)
        lista_eigen_values.append(eigen_values)
    return lista_eigen_values


def orbit_around_Lagrange_points_L2_L4_L5(epsilon, tab_Lagrange, mu, N, delta_t, Scheme) :
    
    Lagrange_points = [1, 3, 4]
    global x_Lagrange
    global y_Lagrange
    
    for number in Lagrange_points :
        
        x_Lagrange = tab_Lagrange[number, 0]
        y_Lagrange = tab_Lagrange[number, 1]

        
        r_init_perfect = np.array([0.5,
                                   0.5,
                                   0, 
                                   0])
        
        r_init_pertubation = np.array([0.5, 
                                       0.5, 
                                       epsilon, 
                                       epsilon])
        
        solution_perfect = integrate_cauchy_problem(r_init_perfect, function_around_Lagrange, delta_t, N, Scheme)
        solution_pertubation = integrate_cauchy_problem(r_init_pertubation, function_around_Lagrange, delta_t, N, Scheme)
        
        plt.figure()
        plt.plot(solution_perfect[0, :], solution_perfect[1, :], label='Perfect Trajectory')
        plt.plot(solution_pertubation[0, :], solution_pertubation[1, :], label='Trajectory with pertubation')
        title = 'Trajectory satellite around Lagrange Point ' + str(number+1)
        plt.title(title)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.plot(0, 0, marker = "o", color = "red", label = 'Lagrange Point ' + str(number+1))
        plt.legend()
        plt.show()
        
        


        
        

###   
delta_t = 0.001
N = int(5*np.pi*1000)
mu = 1.2153*10**(-2)

r_0 = np.array([0.5, 
                0.5, 
                0, 
                0])

r_1 = np.array([0.9, 
                0.5, 
                0, 
                0])

position_m2 = np.array([1-mu, 
                        0.])

position_m1 = np.array([-mu, 
                        0.])
 
solution_3Bproblem_r0 = integrate_cauchy_problem(r_0, function_Kepler_3B, delta_t, N, Runge_Kutta)
solution_3Bproblem_r1 = integrate_cauchy_problem(r_1, function_Kepler_3B, delta_t, N, Runge_Kutta)


##First curves
plt.figure()
plt.plot(solution_3Bproblem_r0[0, :], solution_3Bproblem_r0[1, :], color = 'blue')
plt.title('Trajectory satellite, initial position (0.5, 0.5)')
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(position_m1[0], position_m1[1], marker = "o", color = "red", label = 'Mass 1')
plt.plot(position_m2[0], position_m2[1], marker = "o", color = "black", label = 'Mass 2')
plt.legend()
plt.show()

plt.figure()
plt.plot(solution_3Bproblem_r1[0, :], solution_3Bproblem_r1[1, :], color = 'orange')
plt.title('Trajectory satellite, initial position (0.9, 0.5)')
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(position_m1[0], position_m1[1], marker = "o", color = "red", label = 'Mass 1')
plt.plot(position_m2[0], position_m2[1], marker = "o", color = "black", label = 'Mass 2')
plt.legend()
plt.show()

##Lagrange points
tab_L = Lagrange_points(mu)
print(tab_L)

#Lagrange points plotted
plt.figure()
plt.plot(position_m1[0], position_m1[1], marker = "o", color = "red", label = 'Mass 1')
plt.plot(position_m2[0], position_m2[1], marker = "o", color = "black", label = 'Mass 2')
plt.plot(tab_L[0, 0], tab_L[0, 1], marker = "o", color = "yellow", label = 'L1')
plt.plot(tab_L[1, 0], tab_L[1, 1], marker = "o", color = "green", label = 'L2')
plt.plot(tab_L[2, 0], tab_L[2, 1], marker = "o", color = "purple", label = 'L3')
plt.plot(tab_L[3, 0], tab_L[3, 1], marker = "o", color = "blue", label = 'L4')
plt.plot(tab_L[4, 0], tab_L[4, 1], marker = "o", color = "orange", label = 'L5')
plt.title('Position of Lagrange points')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(-1.7, 1.2)
plt.ylim(-1., 1.7)
plt.legend()
plt.show()

##Stability Lagrange points
lista_eigen_values = stability_lagrange_points(mu, tab_L)
for i in range(len(lista_eigen_values)) :
    array = lista_eigen_values[i]
    print("Eigen values of Lagrange point " + str(i+1) + " : ")
    print(array)   
    

##Orbits around Lagrange points L2, L4 and L5
epsilon = 0.1
orbit_around_Lagrange_points_L2_L4_L5(epsilon, tab_L, mu, int(8*np.pi*1000), delta_t, Runge_Kutta)

#Orbits with initial conditions L1 and L3 
r_L1 = np.array([tab_L[0, 0] + 0.002, 
                  tab_L[0, 1]+0.002, 
                  0, 
                  0])

r_L3 = np.array([tab_L[2, 0]+0.002, 
                  tab_L[2, 1]+0.002, 
                  0, 
                  0])
solution_3B_problem_L1 = integrate_cauchy_problem(r_L1, function_Kepler_3B, delta_t, 5000, Runge_Kutta)
solution_3B_problem_L3 = integrate_cauchy_problem(r_L3, function_Kepler_3B, delta_t, 40000, Runge_Kutta)

plt.figure()
plt.title('Trajectory satellite, initial condition close to L1')
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(solution_3B_problem_L1[0, :], solution_3B_problem_L1[1, :], color = 'red')
plt.plot(position_m1[0], position_m1[1], marker = "o", color = "red", label = 'Mass 1')
plt.plot(position_m2[0], position_m2[1], marker = "o", color = "black", label = 'Mass 2')
plt.plot(tab_L[0, 0], tab_L[0, 1], marker = "o", color = "green", label = 'L2')
plt.legend()
plt.show()

plt.figure()        
plt.title('Trajectory satellite, initial condition close to L3')
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(solution_3B_problem_L3[0, :], solution_3B_problem_L3[1, :], color = 'purple')
plt.plot(position_m1[0], position_m1[1], marker = "o", color = "red", label = 'Mass 1')
plt.plot(position_m2[0], position_m2[1], marker = "o", color = "black", label = 'Mass 2')
plt.plot(tab_L[2, 0], tab_L[2, 1], marker = "o", color = "green", label = 'L3')
plt.legend()
plt.show()   
    
    
    




