#N_body_problem
from numpy import           array, zeros, reshape, shape, linspace ,vstack,pi,append
from numpy.linalg import    norm , eigvals
from scipy.integrate import odeint, LSODA, solve_ivp
import matplotlib.pyplot as plt
from numerical_schemes import Euler_forward, Euler_backward, RungeKutta4, CrankNicholson,LeapFrog
from scipy import optimize
from math import sqrt

#----------------------------------------------------
# Kepler orbit 
#----------------------------------------------------
def Kepler_example(): 

 U0 = array( [1, 0, 0, 1] ) 

 t = linspace(0, 10, 101)

 U = odeint(F_Kepler, U0, t)

 plt.plot(U[:,0], U[:, 1])
 plt.axis('equal')
 plt.grid()
 plt.show()


def F_Kepler(U, t):

    r     =  U[0:2] 
    drdt  =  U[2:4]
    dUdt = array( zeros(4) ) 
   
    dvdt = -r/norm(r)**3 

    dUdt[0:2] = drdt
    dUdt[2:4] = dvdt
    
    return dUdt



#------------------------------------------------------------------
# Orbits of N bodies 
#      U : state vector
#      r, v: position and velocity points to U     
#------------------------------------------------------------------ 
   
def Integrate_NBP():  
    
   def F_NBody(U, t):

       return F_NBody_problem( U, Nb, Nc )

   N =  500    # time steps 
   Nb = 2    # bodies 
   Nc = 2      # coordinates 
   Nt = (N+1) * 2 * Nc * Nb

   t0 = 0; tf = 10#6 * 10 * 3.14 
   Time = linspace(t0, tf, N+1) # array ( [ t0 + (tf -t0 ) * i / N for i in range(N+1) ] ) # eq. fortran Time(0:N) 
 

   U0 = Initial_positions_and_velocities_NBodies( Nc, Nb )
   U_ode = odeint(F_NBody, U0, Time)
   Us_ode  = reshape( U_ode, (N+1, Nb, Nc, 2) ) 
   r_ode   = reshape( Us_ode[:, :, :, 0], (N+1, Nb, Nc) )
   for i in range(Nb):
     plt.plot(  r_ode[:, i, 0], r_ode[:, i, 1] )
   plt.axis('equal')
   plt.legend(['N1', 'N2','N3', 'N4','N5'])
   plt.title(' odeint')
   plt.xlabel('x')
   plt.ylabel('y')
   plt.grid()
   plt.show()

   U0 = Initial_positions_and_velocities_NBodies( Nc, Nb )
   U_n_body_rg4=U0
   for i in range(N):
       tn=Time[i]
       tn1=Time[i+1]
       U=RungeKutta4(U0,tn,tn1,F_NBody)
       U0=U
       U_n_body_rg4=vstack((U_n_body_rg4,U))



   Us_rg4=reshape(U_n_body_rg4, (N+1, Nb, Nc, 2) )
   r_rg4=reshape( Us_rg4[:, :, :, 0], (N+1, Nb, Nc) ) 
   
   
   for i in range(Nb):
     plt.plot(  r_rg4[:, i, 0], r_rg4[:, i, 1] )
   plt.axis('equal')
   plt.legend(['N1', 'N2','N3', 'N4','N5'])
   plt.title(' Runge Kutta 4')
   plt.xlabel('x')
   plt.ylabel('y')
   plt.grid()
   plt.show()



   U0 = Initial_positions_and_velocities_NBodies( Nc, Nb )
   U_n_body_euF=U0
   for i in range(N):
       tn=Time[i]
       tn1=Time[i+1]
       U=Euler_forward(U0,tn,tn1,F_NBody)
       U0=U
       U_n_body_euF=vstack((U_n_body_euF,U))

   Us_euF=reshape(U_n_body_euF, (N+1, Nb, Nc, 2) )
   r_euF=reshape( Us_euF[:, :, :, 0], (N+1, Nb, Nc) ) 
   
   
   for i in range(Nb):
     plt.plot(  r_euF[:, i, 0], r_euF[:, i, 1] )
   plt.axis('equal')
   plt.legend(['N1', 'N2','N3', 'N4','N5'])
   plt.title('Euler Forward')
   plt.xlabel('x')
   plt.ylabel('y')
   plt.grid()
   plt.show()

   U0 = Initial_positions_and_velocities_NBodies( Nc, Nb )
   U_n_body_LF=U0
   k=1
   for i in range(N):
       tn=Time[i]
       tn1=Time[i+1]
       if k==1:
           U1=Euler_forward(U0,tn,tn1,F_NBody)
       
       U=LeapFrog(U1,U0,tn,tn1,F_NBody)
       U0=U1
       U1=U
       U_n_body_LF=vstack((U_n_body_LF,U))
       k=k+1

   Us_LF=reshape(U_n_body_LF, (N+1, Nb, Nc, 2) )
   r_LF=reshape( Us_LF[:, :, :, 0], (N+1, Nb, Nc) ) 
   
   
   for i in range(Nb):
     plt.plot(  r_LF[:, i, 0], r_LF[:, i, 1] )
   plt.axis('equal')
   plt.legend(['N1', 'N2','N3', 'N4','N5'])
   plt.title('Leap-Frog')
   plt.xlabel('x')
   plt.ylabel('y')
   plt.grid()
   plt.show()

   #for i in range(Nb):
   #  plt.plot( Time, r_LF[:, i, 0],  )
   ##plt.axis('equal')
   #  plt.grid()
   #  plt.show()

   U0 = Initial_positions_and_velocities_NBodies( Nc, Nb )
   U_n_body_CN=U0
   for i in range(N):
       tn=Time[i]
       tn1=Time[i+1]
       U=CrankNicholson(U0,tn,tn1,F_NBody)
       U0=U
       U_n_body_CN=vstack((U_n_body_CN,U))

   Us_CN=reshape(U_n_body_CN, (N+1, Nb, Nc, 2) )
   r_CN=reshape( Us_CN[:, :, :, 0], (N+1, Nb, Nc) ) 
   
   
   for i in range(Nb):
     plt.plot(  r_CN[:, i, 0], r_CN[:, i, 1] )
   plt.axis('equal')
   plt.legend(['N1', 'N2','N3', 'N4','N5'])
   plt.title('Crank Nicolson')
   plt.xlabel('x')
   plt.ylabel('y')
   plt.grid()
   plt.show()


   U0 = Initial_positions_and_velocities_NBodies( Nc, Nb )
   U_n_body_euB=U0
   for i in range(N):
       tn=Time[i]
       tn1=Time[i+1]
       U=Euler_backward(U0,tn,tn1,F_NBody)
       U0=U
       U_n_body_euB=vstack((U_n_body_euB,U))

   Us_euB=reshape(U_n_body_euB, (N+1, Nb, Nc, 2) )
   r_euB=reshape( Us_euB[:, :, :, 0], (N+1, Nb, Nc) ) 
   
   
   for i in range(Nb):
     plt.plot(  r_euB[:, i, 0], r_euB[:, i, 1] )
   plt.axis('equal')
   plt.legend(['N1', 'N2','N3', 'N4','N5'])
   plt.title('Euler Backward')
   plt.xlabel('x')
   plt.ylabel('y')
   plt.grid()
   plt.show()
  
#------------------------------------------------------------
#  Initial codition: 6 degrees of freedom per body  
#------------------------------------------------------------
def Initial_positions_and_velocities_NBodies( Nc, Nb ): 
 
    U0 = array( zeros(2*Nc*Nb) )
    U1  = reshape( U0, (Nb, Nc, 2) )  
    r0 = reshape( U1[:, :, 0], (Nb, Nc) )     # position and velocity 
    v0 = reshape( U1[:, :, 1], (Nb, Nc) )
    r0[0,:] = [ 1, 0]
    r0[1,:] = [ -1, 0]
    #r0[2,:]=[ 0, 2]
    #r0[3,:]=[ 0,-2 ]
    #r0[4,:]=[ 0,0]
    v0[0,:] = [ 0, 0.4]
    v0[1,:] = [ 0, -0.4]
    #v0[2,:] = [ 0.4, 0]
    #v0[3,:] = [-0.4,0]
    #v0[4,:] = [0,0]
    return U0 






   
#-----------------------------------------------------------------
#  dvi/dt = - G m sum_j (ri- rj) / | ri -rj |**3, dridt = vi 
#----------------------------------------------------------------- 
def F_NBody_problem(U, Nb, Nc): 
     
 #   Write equations: Solution( body, coordinate, position-velocity )      
     Us  = reshape( U, (Nb, Nc, 2) )  
     F = array( zeros(len(U)) )   
     dUs = reshape( F, (Nb, Nc, 2) )  
     
     r = reshape( Us[:, :, 0], (Nb, Nc) )     # position and velocity 
     v = reshape( Us[:, :, 1], (Nb, Nc) )
     
     drdt = reshape( dUs[:, :, 0], (Nb, Nc) ) # derivatives
     dvdt = reshape( dUs[:, :, 1], (Nb, Nc) )
    
     dvdt[:,:] = 0  # WARNING dvdt = 0, does not work 
    
     for i in range(Nb):   
       drdt[i,:] = v[i,:]
       for j in range(Nb): 
         if j != i:  
           d = r[j,:] - r[i,:]
           dvdt[i,:] = dvdt[i,:] +  d[:] / norm(d)**3 
    
     return F


 # Circular Restricted 3 Body problem 

def CR3BP(U):
    mu = 0.0121505856
    
    x  = U[0];   y = U[1];   z = U[2];
  
    vx = 0;  vy = 0;  vz =0
    #vx = U[4];  vy = U[4];  vz = U[4];
     
    d = sqrt( (x+mu)**2 + y**2 + z**2 ) 
    r = sqrt( (x-1+mu)**2 + y**2 + z**2 ) 
     
    dvxdt = x + 2 * vy - (1-mu) * ( x + mu )/d**3 - mu*(x-1+mu)/r**3
    dvydt = y - 2 * vx - (1-mu) * y/d**3 - mu * y/r**3
    dvzdt = - (1-mu)*z/d**3 - mu*z/r**3
     
    #F =array([ vx, vy, vz, dvxdt, dvydt, dvzdt ] )
    F=array([dvxdt, dvydt, dvzdt ] )

    return F

def CR3BP_2(U,t):
    mu = 0.0121505856
    
    x  = U[0];   y = U[1];   z = U[2];
  
    vx =U[3];  vy =U[4];  vz =U[5]
    
     
    d = sqrt( (x+mu)**2 + y**2 + z**2 ) 
    r = sqrt( (x-1+mu)**2 + y**2 + z**2 ) 
     
    dvxdt = x + 2 * vy - (1-mu) * ( x + mu )/d**3 - mu*(x-1+mu)/r**3
    dvydt = y - 2 * vx - (1-mu) * y/d**3 - mu * y/r**3
    dvzdt = - (1-mu)*z/d**3 - mu*z/r**3
     
    F =array([ vx, vy, vz, dvxdt, dvydt, dvzdt ] )
    

    return F

def CR3BP_3(t,U):
    mu = 0.0121505856
    
    x  = U[0];   y = U[1];   z = U[2];
  
    vx =U[3];  vy =U[4];  vz =U[5]
    
     
    d = sqrt( (x+mu)**2 + y**2 + z**2 ) 
    r = sqrt( (x-1+mu)**2 + y**2 + z**2 ) 
     
    dvxdt = x + 2 * vy - (1-mu) * ( x + mu )/d**3 - mu*(x-1+mu)/r**3
    dvydt = y - 2 * vx - (1-mu) * y/d**3 - mu * y/r**3
    dvzdt = - (1-mu)*z/d**3 - mu*z/r**3
     
    F =array( [vx, vy, vz, dvxdt, dvydt, dvzdt ] )
    

    return F

def System_matrix( F , U0 , t ):

    A=zeros([len(U0), len(U0)])
    eps=1e-6

    for j in range(len(U0)):
        delta=zeros([len(U0)])
        delta[j]=eps
        A[:, j] = ( F( U0 + delta, t ) - F( U0 - delta, t ) )/(2*eps)
    return(A)
        
    

def Integrate_3BP():
    N =  20000    # time steps 
    Nc = 6     # coordinates 
    NL=5 #numero de puntos de lagrange
    #inicialización puntos de Lagrange
    U0=array([[0.8, 0.6 ,0. ,0., 0., 0.],[0.8, -0.6 ,0. ,0., 0., 0.],[ -0.1, 0., 0., 0., 0., 0. ],[ 0.1, 0., 0., 0., 0., 0. ], [ 1.1, 0.0, 0., 0., 0., 0.]]) 

    t0 = 0; tf = 4*pi/0.3 
    Time = linspace(t0, tf, N+1)

    u_lagrange=zeros([5,6])
    for i in range(NL):
        u_lagrange_i= optimize.newton(CR3BP,U0[i,0:3],tol=1.e-016,maxiter=10000,rtol=0.0)
        u_lagrange[i,0:3]=u_lagrange_i

        #print(CR3BP(u_lagrange_i))
    print("Puntos de Lagrange",u_lagrange)


    #para ver la estabilidad de los puntos de Lagrange
    for i in range(NL):
        B = System_matrix(  CR3BP_2,   u_lagrange[i,:], 0.)
        autovalores_i=eigvals(B)
        print('Autovalores_i',autovalores_i)
        

    #integramos cada orbita respecto a un punto cercano al de Lagrange 
    eps_1=10**(-3)
    eps=array([eps_1, eps_1, eps_1, eps_1*0.5, eps_1*0.5, eps_1*0.5])
    
    #Representación puntos de lagrange
    plt.plot(  u_lagrange[0,0], u_lagrange[0,1], 'o' )
    plt.plot( u_lagrange[1,0], u_lagrange[1,1],  'o' )
    plt.plot(u_lagrange[2,0], u_lagrange[2,1] ,  'o')
    plt.plot(u_lagrange[3,0], u_lagrange[3,1] ,  'o')
    plt.plot(u_lagrange[4,0], u_lagrange[4,1] ,  'o')
    plt.plot( 0 , 0, 'o' )
    plt.plot( 1 , 0, 'o')
    plt.text(0,0.1, u'Earth')
    plt.text(1,0.1, u'Moon')
    plt.text( u_lagrange[0,0], u_lagrange[0,1],u'L4')
    plt.text( u_lagrange[1,0], u_lagrange[1,1],  u'L5')
    plt.text(u_lagrange[2,0], u_lagrange[2,1] , u'L3')
    plt.text(u_lagrange[3,0], u_lagrange[3,1] ,  u'L1')
    plt.text(u_lagrange[4,0], u_lagrange[4,1] ,  u'L2')
    plt.title('Lagrange Points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()   
    
    #orbita alrededor de L4 (sale un poco rara) 
    #RungeKutta4
    U0=u_lagrange[0,:]+eps
    print(U0)
    U_n_body_rg4=zeros([N+1,6])
    U_n_body_rg4[0,:]=U0
    for j in range(N):
        tn=Time[i]
        tn1=Time[i+1]
        U=RungeKutta4(U0,tn,tn1,CR3BP_2)
        U0=U
        U_n_body_rg4[j+1,:]=U

    plt.plot( U_n_body_rg4[:,0], U_n_body_rg4[:,1] )
    plt.plot( 0 , 0, 'o')
    plt.plot( 1 , 0, 'o')
    plt.text(0,0.1, u'Earth')
    plt.text(1,0.1, u'Moon')
    plt.plot(  u_lagrange[0,0], u_lagrange[0,1], 'o' )
    plt.text( u_lagrange[0,0], u_lagrange[0,1],u'L4')
    plt.title('L4 Runge Kutta 4')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()

    #odeint
    U0=u_lagrange[0,:]+eps
    U_ode = odeint(CR3BP_2, U0, Time)
    plt.plot( U_ode[:,0], U_ode[:,1] )
    plt.plot( 0 , 0, 'o')
    plt.plot( 1 , 0, 'o')
    plt.text(0,0.1, u'Earth')
    plt.text(1,0.1, u'Moon')
    plt.plot(  u_lagrange[0,0], u_lagrange[0,1], 'o' )
    plt.text( u_lagrange[0,0], u_lagrange[0,1],u'L4')
    plt.title('L4 odeint')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()

    plt.plot( U_ode[:,0], U_ode[:,1] )
    plt.plot(  u_lagrange[0,0], u_lagrange[0,1], 'o' )
    plt.text( u_lagrange[0,0], u_lagrange[0,1],u'L4')
    plt.title('L4 odeint')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()


    #orbita alrededor de L5
    #RungeKutta4
    U0=u_lagrange[1,:]+eps
    print(U0)
    U_n_body_rg4=zeros([N+1,6])
    U_n_body_rg4[0,:]=U0
    for j in range(N):
        tn=Time[i]
        tn1=Time[i+1]
        U=RungeKutta4(U0,tn,tn1,CR3BP_2)
        U0=U
        U_n_body_rg4[j+1,:]=U

    plt.plot( U_n_body_rg4[:,0], U_n_body_rg4[:,1] )
    plt.plot( 0 , 0, 'o')
    plt.plot( 1 , 0, 'o')
    plt.text(0,0.1, u'Earth')
    plt.text(1,0.1, u'Moon')
    plt.title('L5 Runge Kutta 4')
    plt.plot( u_lagrange[1,0], u_lagrange[1,1], 'o' )
    plt.text( u_lagrange[1,0], u_lagrange[1,1],u'L5')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()

    #LSODA
    U0=u_lagrange[1,:]+eps
    U_chimpun =  solve_ivp(CR3BP_3,[Time[0],Time[len(Time)-1]], U0, method='LSODA' )
    plt.plot( U_chimpun.y[0,:], U_chimpun.y[1,:] )
    plt.plot( 0 , 0, 'o')
    plt.plot( 1 , 0, 'o')
    plt.text(0,0, u'Earth')
    plt.text(1,0, u'Moon')
    plt.plot(  u_lagrange[1,0], u_lagrange[1,1], 'o' )
    plt.text( u_lagrange[1,0], u_lagrange[1,1],u'L5')
    plt.title('L5 LSODA')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()

    plt.plot( U_chimpun.y[0,:], U_chimpun.y[1,:] )
    plt.plot(  u_lagrange[1,0], u_lagrange[1,1], 'o' )
    plt.text( u_lagrange[1,0], u_lagrange[1,1],u'L5')
    plt.title('L5 LSODA')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()


    #orbita alrededor de L3
    #RungeKutta4
    U0=u_lagrange[2,:]+eps
    print(U0)
    U_n_body_rg4=zeros([N+1,6])
    U_n_body_rg4[0,:]=U0
    for j in range(N):
        tn=Time[i]
        tn1=Time[i+1]
        U=RungeKutta4(U0,tn,tn1,CR3BP_2)
        U0=U
        U_n_body_rg4[j+1,:]=U

    plt.plot( U_n_body_rg4[:,0], U_n_body_rg4[:,1] )
    plt.plot( 0 , 0, 'o')
    plt.plot( 1 , 0, 'o')
    plt.text(0,0.1, u'Earth')
    plt.text(1,0.1, u'Moon')
    plt.title('L3 Runge Kutta 4')
    plt.plot( u_lagrange[2,0], u_lagrange[2,1], 'o' )
    plt.text( u_lagrange[2,0], u_lagrange[2,1],u'L3')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()

    #odeint
    U0=u_lagrange[2,:]+eps
    U_ode = odeint(CR3BP_2, U0, Time)
    plt.plot( U_ode[:,0], U_ode[:,1] )
    plt.plot( 0 , 0, 'o')
    plt.plot( 1 , 0, 'o')
    plt.text(0,0.5, u'Earth')
    plt.text(1,0.5, u'Moon')
    plt.plot(  u_lagrange[2,0], u_lagrange[2,1], 'o' )
    plt.text( u_lagrange[2,0], u_lagrange[2,1],u'L3')
    plt.title('L3 odeint')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()

    #orbita alrededor de L1
    #RungeKutta4
    U0=u_lagrange[3,:]+eps
    print(U0)
    U_n_body_rg4=zeros([N+1,6])
    U_n_body_rg4[0,:]=U0
    for j in range(N):
        tn=Time[i]
        tn1=Time[i+1]
        U=RungeKutta4(U0,tn,tn1,CR3BP_2)
        U0=U
        U_n_body_rg4[j+1,:]=U

    plt.plot( U_n_body_rg4[:,0], U_n_body_rg4[:,1] )
    plt.plot( 0 , 0, 'o')
    plt.plot( 1 , 0, 'o')
    plt.text(0,0, u'Earth')
    plt.text(1,0, u'Moon')
    plt.plot( u_lagrange[3,0], u_lagrange[3,1], 'o' )
    plt.text( u_lagrange[3,0], u_lagrange[3,1],u'L1')
    plt.title('L1 Runge Kutta 4')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()

    #LSODA
    U0=u_lagrange[3,:]+eps
    U_chimpun =  solve_ivp(CR3BP_3,[Time[0],Time[len(Time)-1]], U0, method='LSODA' )
    plt.plot( U_chimpun.y[0,:], U_chimpun.y[1,:] )
    plt.plot( 0 , 0, 'o')
    plt.plot( 1 , 0, 'o')
    plt.text(0,0, u'Earth')
    plt.text(1,0, u'Moon')
    plt.plot(  u_lagrange[3,0], u_lagrange[3,1], 'o' )
    plt.text( u_lagrange[3,0], u_lagrange[3,1],u'L1')
    plt.title('L1 LSODA')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()

    #orbita alrededor de L2 
    #RungeKutta4
    U0=u_lagrange[4,:]+eps
    print(U0)
    U_n_body_rg4=zeros([N+1,6])
    U_n_body_rg4[0,:]=U0
    for j in range(N):
        tn=Time[i]
        tn1=Time[i+1]
        U=RungeKutta4(U0,tn,tn1,CR3BP_2)
        U0=U
        U_n_body_rg4[j+1,:]=U

    plt.plot( U_n_body_rg4[:,0], U_n_body_rg4[:,1] )
    plt.plot( 0 , 0, 'o')
    plt.plot( 1 , 0, 'o')
    plt.text(0,0.5, u'Earth')
    plt.text(1,0.5, u'Moon')
    plt.plot( u_lagrange[4,0], u_lagrange[4,1], 'o' )
    plt.text( u_lagrange[4,0], u_lagrange[4,1],u'L2')
    plt.title('L2 Runge Kutta 4')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()

    #odeint
    U0=u_lagrange[4,:]+eps
    U_ode = odeint(CR3BP_2, U0, Time)
    plt.plot( U_ode[:,0], U_ode[:,1] )
    plt.plot( 0 , 0, 'o')
    plt.plot( 1 , 0, 'o')
    plt.text(0,0.5, u'Earth')
    plt.text(1,0.5, u'Moon')
    plt.plot(  u_lagrange[4,0], u_lagrange[4,1], 'o' )
    plt.text( u_lagrange[4,0], u_lagrange[4,1],u'L2')
    plt.title('L2 odeint')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()

    #Arenstof orbit 
    t=linspace(0,17,10000)
    res=odeint(rhs,[0.994,0.0,0.0,-2.00158510637908],t)
    y1r,y2r,y3r,y4r=res[:, 0],res[:, 1],res[:, 2],res[:, 3]
    plt.plot(y1r,y2r)
    plt.plot( 0 , 0, 'o' )
    plt.plot( 1 , 0, 'o')
    plt.text(0,0.1, u'Earth')
    plt.text(0.9,0.1, u'Moon')
    plt.xlabel('x')
    plt.ylabel('y')    
    plt.show()

#Arenstof orbit 
def rhs(u,t):
    y1,y2,y3,y4 = u
    a=0.012277471; b=1.0-a;    
    D1=((y1+a)**2+y2**2)**(3.0/2);
    D2=((y1-b)**2+y2**2)**(3.0/2);
    res = [y3,\
           y4,\
           y1+2.0*y4-b*(y1+a)/D1-a*(y1-b)/D2, \
           y2-2.0*y3-b*y2/D1-a*y2/D2
           ]
    return res




#Integrate_NBP()
Integrate_3BP()
