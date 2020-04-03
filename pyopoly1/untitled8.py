# Demonstration of constructing interpolants on Gauss quadrature nodes

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
from families import HermitePolynomials
import variableTransformations as VT
import indexing
import opolynd




# Font styling
rcParams['font.family'] = 'serif'
rcParams['font.weight'] = 'bold'
fontprops = {'fontweight': 'bold'}

# Test function
u = lambda xx: xx
def u(points):
    return np.ones(len(points))

d = 2
k = 40

H = HermitePolynomials()

ab = H.recurrence(k+1)

N = 10
x = np.linspace(-1,1, N)

X,Y = np.meshgrid(x,x)
XX = np.concatenate((X.reshape(X.size,1), Y.reshape(Y.size,1)), axis=1)
scaling = np.asarray([[0, 1], [0,1]])
xCan=VT.map_to_canonical_space(XX, 2, scaling)

lambdas = indexing.total_degree_indices(d, k)
V = opolynd.opolynd_eval(xCan, lambdas, ab, H)
Vinv = np.linalg.inv(V[:,:np.size(V,0)])

ii = np.matmul(Vinv,V[:,:np.size(V,0)])
# Compute interpolant
c = np.matmul(Vinv, u(xCan))


# Zinterp_eval = np.dot(V[:,:np.size(V,0)], c)


# plt.figure()
# lines = []
# lines.append(plt.plot(xg, u(xg))[0])
# lines.append(plt.plot(xg, interp_eval, '.')[0])

# plt.xlabel(r'$x$', **fontprops)
# plt.title(r'Plot of $u$ and $N$-point interpolants', **fontprops)

