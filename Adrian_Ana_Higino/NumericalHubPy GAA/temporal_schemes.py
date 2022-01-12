#   ------------------------------------------------------------------
#                           TEMPORAL SCHEMES 
#                    AMPLIACIÓN DE MATEMÁTICAS 1
#               MÁSTER UNIVERSITARIO DE SISTEMAS ESPACIALES
#   ------------------------------------------------------------------



from scipy.optimize import newton


# Euler
def Euler(t1, t2, U1, F):

    Dt = t2 - t1
    U2 = U1 + Dt * F( U1, t1 )

    return U2


# Runge Kutta 4
def RK4(t1, t2, U1, F):

    Dt = t2 - t1



    k1 = F(U1, t1)
    k2 = F(U1 + k1*(Dt/2), t1 + (Dt/2))
    k3 = F(U1 + k2*(Dt/2), t1 + (Dt/2))
    k4 = F(U1 + k3*Dt, t1 + Dt)



    U2 = U1 + (k1 + 2*k2 + 2*k3 + k4)*Dt / 6



    return U2


# Crank-Nicolson
def func_i_CN(U2, U1, t1, t2, F):

 

    Dt= t2 - t1

 

    return( U2 - U1 - (Dt/2.) * (F(U2, t2) + F(U1, t1) ) )

 


def CN(t1, t2, U1, F):

 

    Dt = t2 - t1

 

    U2= newton(func_i_CN, U1, fprime=None, args=(U1, t1, t2, F), maxiter= 1000)
    return(U2)