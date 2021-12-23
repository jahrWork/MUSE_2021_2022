import scipy.optimize as so

def Euler(U_prev, F, delta_t, n) :
    t_n = int(n*delta_t)
    U_next = U_prev + F(t_n, U_prev)*delta_t

    return U_next

def Euler_inverse(U_prev, F, delta_t, n):
    t_n = int(n*delta_t)
    return so.newton(lambda X : X - U_prev - delta_t*F(t_n, X), U_prev, tol = 10e-8, maxiter = 10000)

def Crank_Nicholson(U_prev, F, delta_t, n) :
    t_n = int(n*delta_t)
    return so.newton(lambda X : X - U_prev - 0.5*delta_t*(F(t_n, U_prev) + F(t_n, X)), U_prev, tol = 10e-6, maxiter = 10000)

def Runge_Kutta(U_prev, F, delta_t, n) :
    t_n = int(n*delta_t)
    k1 = F(t_n, U_prev)
    k2 = F(t_n + 0.5*delta_t, U_prev + delta_t*0.5*k1)
    k3 = F(t_n + 0.5*delta_t, U_prev + delta_t*0.5*k2)
    k4 = F(t_n + delta_t, U_prev + delta_t*k3)
    return U_prev + (1/6)*delta_t*(k1 + 2*k2 + 2*k3 + k4)

def Leap_Frog(U_n, F, delta_t, n, U_n_1 = 'zero') :
    
    t_n = int(n*delta_t)
    if isinstance(U_n_1, str) :
        return U_n + delta_t*F(t_n, U_n)
    else :
        return U_n_1 + 2*delta_t*F(t_n, U_n)
