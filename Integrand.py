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
<<<<<<< HEAD
        val = np.log(np.max(G[0,:] * phat[0] + G[-1,:] * phat[-1]))
=======
        val = np.log(max(val1, val2))
>>>>>>> NewRowToGMethodStart
        return val


def computeIntegrandArray(G, phat):
    val = np.zeros(len(G))
    for w in range(len(G)):
        val[w] = np.max(G[w,:]*phat[w])
    return val


def calculateIntegrand(G, phat):
<<<<<<< HEAD
    val = np.zeros([phat.size, phat.size])
    for i in range(np.size(phat)):
        val[i,:]= G[i,:] * phat[i]
=======
    val = np.zeros([np.size(G, 0), np.size(G, 1)])
    for i in range(np.size(G, 1)):
        val[:,i]= G[i,:] * phat[i]
    r = np.sum(val,0)
>>>>>>> NewRowToGMethodStart
    return val