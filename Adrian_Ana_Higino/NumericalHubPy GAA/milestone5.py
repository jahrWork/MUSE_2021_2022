#   ------------------------------------------------------------------
#                           Milestone #5
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
from stabreg import stabreg
from nbody import *

def M5():
	NBP()
