import numpy as np
def REA(z):
    return 1 + z
def REI(z):
    return 1/(1 - z)
def RCN(z):
    return (1 + z/2)/(1 - z/2)
def RRK(z):
    d = 4
    m = np.zeros(d + 1, dtype=complex)
    for k in np.arange(0, d + 1):
        m[k] = (1/np.math.factorial(k))*z**k
    return(np.sum(m))
def RLF(z):
    return(max([abs(z + np.sqrt(1 + z**2)), abs(z - np.sqrt(1 + z**2))]))