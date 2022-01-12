#   ------------------------------------------------------------------
#                           N-BODIES PORBLEM
#                    AMPLIACIÓN DE MATEMÁTICAS 1
#               MÁSTER UNIVERSITARIO DE SISTEMAS ESPACIALES
#   ------------------------------------------------------------------

# Imports
from numpy import               array, linspace, pi, zeros, reshape, shape
from numpy.linalg import        norm
from kepler_orbit import        F_kep
from cauchyproblem import       CauchyProblemSol
from temporal_schemes import    Euler, RK4, CN
import matplotlib.pyplot as     plt
from stabreg import             stabreg
from scipy.integrate import     odeint


def NBP():  
    
   def F_NB(U, t):

       return F_NBP (U, Nb, Nc )

   N =  500    # time steps 
   Nb = 2      # bodies 
   Nc = 2      # coordinates 
   Nt = (N+1) * 2 * Nc * Nb

   t0 = 0; tf = 6 * 10 * 3.14 
   Time = linspace(t0, tf, N+1)
 
   U0 = init_con( Nc, Nb )
 
   U = odeint(F_NB, U0, Time)

   Us  = reshape( U, (N+1, Nb, Nc, 2) ) 
   r   = reshape( Us[:, :, :, 0], (N+1, Nb, Nc) ) 
   
   for i in range(Nb):
     plt.plot(  r[:, i, 0], r[:, i, 1] )
     plt.axis('equal')
     plt.grid()
     plt.show()
  
def init_con( Nc, Nb ): 
 
    U0 = array( zeros(2*Nc*Nb) )
    U1  = reshape( U0, (Nb, Nc, 2) )  
    r0 = reshape( U1[:, :, 0], (Nb, Nc) )     
    v0 = reshape( U1[:, :, 1], (Nb, Nc) )
    r0[0,:] = [ 1, 0]
    r0[1,:] = [ -1, 0]
    v0[0,:] = [ 0, 0.4]
    v0[1,:] = [ 0, -0.4]

    return U0 
     
 
def F_NBP(U, Nb, Nc): 
     
 #   Write equations: Solution( body, coordinate, position-velocity )      
     Us  = reshape( U, (Nb, Nc, 2) )  
     F = array( zeros(len(U)) )   
     dUs = reshape( F, (Nb, Nc, 2) )  
     
     r = reshape( Us[:, :, 0], (Nb, Nc) )     # position and velocity 
     v = reshape( Us[:, :, 1], (Nb, Nc) )
     
     drdt = reshape( dUs[:, :, 0], (Nb, Nc) ) # derivatives
     dvdt = reshape( dUs[:, :, 1], (Nb, Nc) )
    
     dvdt[:,:] = 0 
    
     for i in range(Nb):   
       drdt[i,:] = v[i,:]
       for j in range(Nb): 
         if j != i:  
           d = r[j,:] - r[i,:]
           dvdt[i,:] = dvdt[i,:] +  d[:] / norm(d)**3 
    
     return F