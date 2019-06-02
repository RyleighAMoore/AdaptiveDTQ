# Demonstration of constructing interpolants on Gauss quadrature nodes

import numpy as np
from matplotlib import pyplot as plt
from polynomial_families import JacobiPolynomials


# Interpolates pdf at zeros of orthogonal polynomials
# and then returns the values of the PDF at x+f(x)h
def interpolate(x, u, f, h, size):
    # Jacobi polynomials
    alpha = 0.
    beta = 0.
    J = JacobiPolynomials(alpha=alpha, beta=beta)

    # Grid for computing errors
    N = size

    # Compute interpolation grid
    x, w = J.gauss_quadrature(N)

    # Compute interpolant
    c = np.dot(J.eval(x, range(N)).T, w * u)
    Verr = J.eval(x, range(np.max(N)))
    interp_eval = np.dot(Verr, c)

    # plt.figure()
    # plt.plot(x, interp_eval, '.')
    # plt.plot(x, u)
    # plt.show()

    xs = x-f(x)*h
    Verr2 = J.eval(xs, range(N))
    interp_eval2 = np.dot(Verr2, c)
    interp_eval_final = lambda xx: interp_eval2*((xx > -1) & (xx<1)) + 0*(abs(xx)<=1)
    # plt.figure()
    # plt.plot(xs, interp_eval2, '.')
    # plt.plot(x, u)
    # plt.show()
    final = interp_eval_final(xs)
    return final, x


