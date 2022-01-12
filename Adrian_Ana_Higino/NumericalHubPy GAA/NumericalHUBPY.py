#   ------------------------------------------------------------------
#                         Numerical HUBPY
#                    AMPLIACIÓN DE MATEMÁTICAS 1
#               MÁSTER UNIVERSITARIO DE SISTEMAS ESPACIALES
#   ------------------------------------------------------------------


# Imports
from numpy import array, linspace, pi
from kepler_orbit import F_kep
from cauchyproblem import CauchyProblemSol
from temporal_schemes import Euler, RK4, CN
from temporalerror import Extrapolacion_Richardson, Velocidad_Convergencia
import matplotlib.pyplot as plt
from milestone2 import M2
from milestone3 import M3
from milestone4 import M4
from milestone5 import M5


# Main Numerical HUBPY

print ('Welcome to Numerical HUBPY')
print ('Select an option')
print ('*******************************************************')
print ('1. Kepler orbit')
print ('2. Error in orbits')
print ('3. Stability Region')
print ('4. N-Bodies Problem')
print ('5. Lagrange Points')
print ('*******************************************************')
option=input("Please, select an option:")

if option == "1": M2()
if option ==  "2": M3()
if option == "3": M4()
if option == "4": M5()
else :
    print ('Sorry, we are still working on this')


