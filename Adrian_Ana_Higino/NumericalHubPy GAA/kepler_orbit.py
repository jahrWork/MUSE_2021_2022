#   ------------------------------------------------------------------
#                            KEPLER_ORBIT
#                    AMPLIACIÓN DE MATEMÁTICAS 1
#               MÁSTER UNIVERSITARIO DE SISTEMAS ESPACIALES
#   ------------------------------------------------------------------




# Imports
from numpy import zeros, array, pi, concatenate
from numpy.linalg import norm


# Main Kepler Orbit

    # Force of the Kepler movement: F_Kepler
def F_kep2(U, t):
       # here, U is a vector
       r = U[0:2]
       drdt = U[2:4]

       #F_kep = [drdt,- r/(norm(r)**3)]
       F_kep = ( drdt, - r/(norm(r)**3) )

       return F_kep

   # Force of the Kepler movement: F_Kepler
def F_kep(U, t):

    # here, U is a vector
    # split
    r = U[0:2]
    drdt = U[2:4]
    F_kep = concatenate( [drdt,- r/(norm(r)**3)] )

    return F_kep
    
def oscillator( U, t ):

    return array([ U[1], -U[0] ])



