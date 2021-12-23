import numpy as np
from Schemes import *

def integrate_cauchy_problem(X_0, F, delta_t, n, Scheme) :

    X_prev = X_0
    Solution = np.zeros((len(X_prev), n+1))
    Solution[:, 0] = X_0
    
    Solution[:, 1] = Scheme(X_prev, F, delta_t, n)
    
    X_prev = Solution[:, 1]
        
    for k in range(1, n) :
        
        if Scheme != Leap_Frog :
            Xf = Scheme(X_prev, F, delta_t, n)
            Solution[:, k+1] = Xf
            X_prev = Xf
        else :
            X_n_1 = Solution[:, k-1]
            Xf = Scheme(X_prev, F, delta_t, n, X_n_1)
            Solution[:, k+1] = Xf
            X_prev = Xf
    
    return Solution




