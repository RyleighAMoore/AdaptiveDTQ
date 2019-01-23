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
        val = np.log(np.max(G[0,:] * phat[0] + G[-1,:] * phat[-1]))
        return val


def calculateIntegrand(G, phat):
    te = np.size(G,0)
    w = phat.size
    ee = np.size(G, 1)
    val = np.zeros([np.size(G,0), np.size(G,1)])

    for i in range(ee):
        r = phat[i]
        t = G[i, :]
        val[i,:]= G[i,:] * phat[i]
    return val