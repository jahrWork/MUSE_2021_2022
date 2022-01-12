#   ------------------------------------------------------------------
#                           Milestone #4
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


def M4():

	############################
	Temporal_Scheme = RK4
	x = linspace(-5, 5, 100)
	y = linspace(-5, 5, 100)
	############################

	rho = stabreg(Temporal_Scheme, x, y)