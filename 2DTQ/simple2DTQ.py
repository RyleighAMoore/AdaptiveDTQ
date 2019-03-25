import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def f(x):
    return x


def g(x):
    return x


def G(x1, x2, y1, y2, h):
    1 / (2 * np.pi * h ** 2 * np.sqrt(g(y1) ** 2) * np.sqrt(g(y2) ** 2)) * np.exp(
        -0.5 * ((((x1 - (y1 + f(y1) * h)) ** 2) / h * g(y1) ** 2) + ((x2 - (y2 + f(y2) * h)) ** 2) / h * g(y2) ** 2))


T = 1  # final time, code computes PDF of X_T
s = 0.75  # the exponent in the relation k = h^s
h = 0.01  # temporal step size
init = 0  # initial condition X_0
numsteps = int(np.ceil(T / h))

assert numsteps > 0, 'The variable numsteps must be greater than 0'

# define spatial grid
k = h ** s
xMin = -4
xMax = 4

x1 = np.arange(xMin, xMax, k)
x2 = np.arange(xMin, xMax, k)