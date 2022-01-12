#   ------------------------------------------------------------------
#                           Milestone #2
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

def M2():

	############################
	U0 = array( [1, 0, 0, 1])
	N = 10000
	Time = linspace( 0, 12*pi, N )
	U = CauchyProblemSol( Time, U0, RK4, F_kep )
	############################

	plt.plot(U[:, 0], U[:, 1], color ='b')
	plt.axis('square')
	plt.xlabel("x position", size = 12)
	plt.ylabel("y position", size = 12)
	plt.title("Phase Diagram for Kepler Orbit", size = 16)
	plt.grid(True)
	plt.show()
