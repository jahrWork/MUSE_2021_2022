#   ------------------------------------------------------------------
#                           STABILITY REGIONS
#                    AMPLIACIÓN DE MATEMÁTICAS 1
#               MÁSTER UNIVERSITARIO DE SISTEMAS ESPACIALES
#   ------------------------------------------------------------------

from cauchyproblem import CauchyProblemSol
from kepler_orbit import F_kep
from temporal_schemes import Euler, RK4, CN
from numpy import array, linspace, pi, zeros, log, vstack, ones, polyfit, transpose, abs, linspace
from numpy.linalg import norm, lstsq
import matplotlib.pyplot as plt

def stabreg(Temporal_Scheme, x, y):

	N = len(x)
	M = len(y)
	rho = zeros([N, M])
	t, dt = 0, 1

	for i in range(N):
		for j in range(M):
			w = complex(x[i], y[j])
			U1 = complex(1, 0)
			r = Temporal_Scheme(U1, t, dt, lambda U, t : w*U)
			rho[i,j] = abs(r)

	plt.contour( x, y, rho, linspace( 0., 1., 11 ))
	plt.show()

	return rho



