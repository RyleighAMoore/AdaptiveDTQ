# Demonstration of Gauss quadrature with orthogonal polynomials

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
from families import JacobiPolynomials, HermitePolynomials

# Font styling
rcParams['font.family'] = 'serif'
rcParams['font.weight'] = 'bold'
fontprops = {'fontweight': 'bold'}

N = 100

# Jacobi polynomials
alpha = 0.
beta = 0.
J = JacobiPolynomials(alpha=alpha, beta=beta)
x,w = J.gauss_quadrature(N)

# This quadature rule can exactly integrate polynomials up to degree
# 2N-1
V = J.eval(x, range(2*N+1))
exact_integrals = np.zeros(2*N+1)
exact_integrals[0] = J.recurrence(1)[0,1]**2
computed_integrals = np.dot(w.T, V)

errors = np.abs(exact_integrals - computed_integrals)

plt.semilogy(range(2*N+1), errors)
plt.xlabel(r'$n$', **fontprops)
plt.title(r'Quadrature error (${0:d}$-point) for integrating $p_n$'.format(N), **fontprops)

# Form Gramian of orthonormal polynomials
# Quadrature rule integrates up to degree 2N-1 ===> it integrates
# products of polynomials up to degree (N-1)
V = J.eval(x, range(N))
G = np.dot(w*V.T, V)

plt.matshow(G)
plt.colorbar()
plt.xlabel(r'$j$', **fontprops)
plt.ylabel(r'$k$', **fontprops)
plt.title(r'${0:d}$-point Gauss quadrature applied to $p_j p_k$'.format(N), **fontprops)

plt.show()