import numpy as np
import matplotlib.pyplot as plt

# This function is used to determine if more rows are needed in the kernel matrix G.
# Basically, it checks that we are maintaining zero at the boundaries of the integrand.
def computeEpsilon(G, phat):
    tol = 0
    val1 = np.max(G[0,:]*phat[0])
    val2 = np.max(G[-1,:]*phat[-1])
    if (val1 <= tol) & (val2 <= tol):
        return -np.inf
    else:
        val = np.log(val1 + val2)
        return val


def computeEpsilonArray(G, phat):
    val = np.zeros(len(G))
    for w in range(len(G)-1):
        val[w] = np.max(G[w,:]*phat[w])
    return val


def calculateIntegrand(G, phat):
    val = np.zeros([np.size(G, 0), np.size(G, 1)])
    for i in range(np.size(G, 1)):
        val[:,i]= G[i,:] * phat[i]
    return val