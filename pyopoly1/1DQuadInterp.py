# Demonstration of constructing interpolants on Gauss quadrature nodes

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
from families import HermitePolynomials
import variableTransformations as VT


# Font styling
rcParams['font.family'] = 'serif'
rcParams['font.weight'] = 'bold'
fontprops = {'fontweight': 'bold'}

# Test function
u = lambda xx: xx**4

N=4

H = HermitePolynomials()

xg, wg = H.gauss_quadrature(N)

sigma = .1
mu = 0
scaling = np.asarray([[mu, sigma]])

xCan=VT.map_to_canonical_space(xg, 1, scaling)

V = H.eval(xg, range(np.max(N)))

# Compute interpolation grid
# x,w = H.gauss_quadrature(N)

#Ryleigh
Vinv = np.linalg.inv(V)
# test = np.matmul(V,Vinv)
# test2 = np.matmul(Vinv,V)

c = np.matmul(Vinv, u(xg))

# Compute interpolant

interp = np.dot(V,c)


plt.figure()
lines = []
lines.append(plt.plot(xg, u(xg))[0])
lines.append(plt.plot(xg, interp, '.')[0])

# plt.xlabel(r'$x$', **fontprops)
# plt.title(r'Plot of $u$ and $N$-point interpolants', **fontprops)

