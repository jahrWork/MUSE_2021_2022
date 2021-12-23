#### N-BODY PROBLEM OPTIMIZED AND NO POINTERS IMPLEMENTATION ####
"""
Created on Tue Dec 21 16:23:55 2021
@author: Dan, Matis, Victor


"""
#### N-BODY PROBLEM OPTIMIZED WITH POINTERS ####

from numpy import           array, zeros, reshape, shape, linspace   
from numpy.linalg import    norm
from scipy.integrate import odeint
from time import process_time



import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import os 
from scipy.optimize import curve_fit
import datetime
from scipy.interpolate import interp1d 
import math
import seaborn as sns
from sklearn.metrics import r2_score
import numba


#TEMPORAL SCHEMES


def Cauchy_problem(F,t,U0, Temporal_scheme):
    
    start= process_time()
    
    N, Nv=len(t)-1,len(U0)
    U= array(zeros([N+1, Nv]))
    U[0,:]= U0
    
    for n in range(N):
        
        U[n+1,:]= Temporal_scheme(U[n,:], t[n+1]-t[n], t[n], F)

    finish= process_time()   
    print("Cauchy_Problem, CPU Time", finish-start," seconds.")
    return U



def RK4(U,dt,t,F):
    
       
        k1= F(U,t)
        k2= F(U+np.multiply(dt,np.multiply(k1,0.5)),t+dt/2)
        k3= F(U+np.multiply(dt,np.multiply(k2,0.5)),t+dt/2)
        k4= F(U+np.multiply(dt,k3),t+dt)
    
        return U+dt*( k1 + np.multiply(2,k2) + np.multiply(2,k3) + k4 )/6
#------------------------------------------------------------------
# Orbits of N bodies 
#      U : state vector
#      r, v: position and velocity points to U     
#------------------------------------------------------------------   

def Integrate_NBP():  
    

   def F_NBody(U, t):

       return F_NBody_problem( U, Nb, Nc )
 
  

   N =  1000  # time steps 
   Nb = 40  # bodies 
   Nc = 3      # coordinates 
   

   t0 = 0; tf = 1
   Time = linspace(t0, tf, N+1)  
 
   U0 = Initial_positions_and_velocities(Nb,Nc)

   
   start= process_time()
   
  
   U = odeint(F_NBody, U0, Time)
    # U = Cauchy_problem(F_NBody,Time, U0, RK4)
    
   finish= process_time()   
   print("CPU Time", finish-start," seconds.")
   print(np.size(U))
   
   U=U.reshape(Nb*(N+1),2*Nc)  #from array vector (1d) to array (2d) (for plotting)
  


   #PLOTTING OF STATE VECTOR 
   
   for t in range(np.size(Time)):
        plt.scatter(U[t*Nb:(t*Nb + Nb),0],U[t*Nb:(t*Nb + Nb),1])
        plt.axis('equal')
        plt.title('Position of N bodies in 1 second') 
        plt.grid()
        plt.show()
   
    
   #SIMULATION OF STATE VECTOR. TO RUN THE SIMULATION ENSURE THAT IT OPENS IN THE CONSOLE.
   # A GOOD VISUALIZATION OF THE SIMULATION WOULD BE 500 STEPS WITH 100 BODIES
    
   # for t in range(N):
                  
   #        plt.scatter( r[t, :, 0], r[t, :, 1] )
   #        plt.scatter(r[max(0,t-50):t, :, 0], r[max(0,t-50):t, :, 1],s=1,color=[.7,.7,1])
   #        plt.axis('equal')
   #        plt.xlim(-10,10)
   #        plt.ylim(-10,10)
   #        plt.grid()
   #        plt.show()
        
        
#------------------------
#------------------------------------------------------------
#  Initial codition: 6 degrees of freedom per body  
#------------------------------------------------------------

def Initial_positions_and_velocities(Nb,Nc): 
 
    U0 = array(zeros((Nb,2*Nc))) # position and velocity 
    
    
    np.random.seed(15) #seed is used to get the same random initial conditions per compilation 
    position  = np.random.randn(int(Nb/2),Nc)
    np.random.seed(10)
    velocity= np.random.randn(int(Nb/2),Nc)

    #to conserve the amount of movement:
        
    #Nb/2 bodies
    for i in range(int(Nb/2)):
        U0[i,0:Nc]= position[i]
        U0[i,Nc:2*Nc]= velocity[i]
    #Nb/2 to Nb bodies
    n=0
    for i in range(int(Nb/2),Nb):
        U0[i,0:Nc]= -position[n]
        U0[i,Nc:2*Nc]= -velocity[n]
        n=n+1
        
    U0=U0.flatten() #from array (2d) to array vector (1d)
    return U0
     
#-----------------------------------------------------------------
#  dvi/dt = - G m sum_j (ri- rj) / | ri -rj |**3, dridt = vi 
#----------------------------------------------------------------- 

#@numba.jit
def F_NBody_problem(U, Nb, Nc): 
    
     #G=6.673E-11    #Newton's gravitational constant
     G=1
     softening= 0.1 #To avoid collisions between bodies
     
     np.random.seed(9)
     mass_n = np.random.rand(Nb,1)
     
     mass_matrix = mass_n*np.ones((Nb, Nb))
     np.fill_diagonal(mass_matrix,0)
    
     U=U.reshape(Nb,2*Nc)  #from array vector(1d) to array (2d)
     

     r=U[:,0:Nc]                          #position
     v=U[:,Nc:2*Nc]                       #velocity
     

     dvdt=array(zeros((Nb,Nc)))          #zeros of derivates
     drdt=array(zeros((Nb,Nc)))
     F=array(zeros((Nb,2*Nc)))
     
     for i in range(Nb):   
          
          drdt[i,:] = v[i,:]
          d = r - r[i,:]
          norma_d = norm(d,axis=1)

          dvdt[i,:] = G*np.sum(np.transpose(d)*((mass_matrix[:,i])/((norma_d+softening)**3)),axis=1)
          
          ###using spheres of influence:
          
          # for j in range(Nb): 
          #   r_sphere=(mass_n[i]/mass_n[j])**(2/5)
          #   d = r[j,:] - r[i,:]
          #   d_norm = norm(d)
            
          #   if j != i:
                
          #     dvdt[i,:] = dvdt[i,:] + ((1.0 * mass_matrix[i,j] * index[j,:]) / ((norma_index[j]+softening)**3))
              #dvdt[i,:] = dvdt[i,:] + ((G * mass_n[i]*mass_n[j] * d) / norm(d+softening)**3)

     F[:,0:Nc]=drdt[:,:]
     F[:,Nc:2*Nc]=dvdt[:,:]
     F=F.flatten() #from array (2d) to array vector (1d)
     
     return F

     



Solucion= Integrate_NBP()




#norm(d)<r_sphere and 












 

  
