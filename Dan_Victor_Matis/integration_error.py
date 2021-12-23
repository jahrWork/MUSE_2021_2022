import numpy as np
from Integrate_Cauchy_problem import integrate_cauchy_problem
from Schemes import Euler, Euler_inverse, Crank_Nicholson, Runge_Kutta

def integration_error(X_0, F, delta_t, n, Scheme) :
    Uf = integrate_cauchy_problem(X_0, F, delta_t, n, Scheme)
    Vf = integrate_cauchy_problem(X_0, F, delta_t/2, 2*n, Scheme)
    
    q = 0
    
    if (Scheme == Euler) or (Scheme == Euler_inverse):
        q = 1
    elif Scheme == Crank_Nicholson:
        q = 2
    elif Scheme == Runge_Kutta :
        q = 4
        
        
    Error = np.zeros((len(X_0), n+1))
    for k in range(n+1) :
        Error[:, k] = (Uf[:, k] - Vf[:, 2*k])/(1-0.5**q) 
    
    return Error

def rate_convergence(X_0, F, delta_t, number_points, Scheme) :
    log_E = np.zeros(number_points)
    log_N = np.zeros(number_points)
    
    for i in range(number_points) :
        N = int(10**((i+1)/2))
        
        log_E[i] = np.log10(np.linalg.norm((integration_error(X_0, F, delta_t, N, Scheme))))
        log_N[i] = np.log10(N)
        
    return log_N, log_E
        
    
        