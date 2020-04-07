# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 18:51:19 2020

@author: Rylei
"""


# Demonstration that preconditioned interpolation matrices on Gauss
# quadrature nodes are well-conditioned and don't require inversion.

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
from families import HermitePolynomials

# Font styling
rcParams['font.family'] = 'serif'
rcParams['font.weight'] = 'bold'
fontprops = {'fontweight': 'bold'}

H = HermitePolynomials()

# Test function
nval = 5 # Degree nval-1 polynomial
u = lambda xx: H.eval(xx, nval-1)[:,0]

N = np.arange(nval, 50,dtype=int)

# Allocation
V_cond = np.zeros(N.size)
wV_cond = np.zeros(N.size)
V_err = np.zeros(N.size)
wV_err = np.zeros(N.size)

for (ind_n, n) in enumerate(N):

    # Compute interpolation grid
    x,w = H.gauss_quadrature(n)

    V = H.eval(x, range(n))
    wV = (np.sqrt(w)*V.T).T

    # Direct inversion
    c1 = np.dot(np.linalg.inv(V), u(x))
    # Quadrature method
    c2 = np.dot(wV.T, np.sqrt(w)*u(x))

    # The exact solution is a vector c with c[nval] = 1, and zeros
    # elsewhere
    c1[nval-1] -= 1.
    c2[nval-1] -= 1.
    V_err[ind_n] = np.linalg.norm(c1)
    wV_err[ind_n] = np.linalg.norm(c2)

    V_cond[ind_n] = np.linalg.cond(V)
    wV_cond[ind_n] = np.linalg.cond(wV)

plt.subplot(1,2,1)
line1 = plt.semilogy(N, V_cond)[0]
line2 = plt.semilogy(N, wV_cond)[0]
plt.legend((line1, line2), (r'$\kappa(V)$', r'$\kappa(\sqrt{W} V)$'))
plt.xlabel(r'$N$', **fontprops)
plt.ylabel(r'$\kappa(V)$', **fontprops)
plt.title(r'Interpolation matrix condition numbers', **fontprops)

plt.subplot(1,2,2)
line1 = plt.semilogy(N, V_err)[0]
line2 = plt.semilogy(N, wV_err)[0]
plt.legend((line1, line2), (r'$u \mapsto V^{-1} u$', r'$u \mapsto V^T W u$'))
plt.xlabel(r'$N$', **fontprops)
plt.ylabel(r'$L^2_w$ interpolant error', **fontprops)
plt.title(r'$N$-point interpolation errors using different methods', **fontprops)

plt.show()
