import numpy as np
import GMatrix
import Functions as fun

def setUpNonCorrectedGrid(xMin, xMax, k, a, b, h):
    xvec = np.arange(xMin, xMax, k)
    phat = fun.dnorm(xvec, a, b)  # pdf after one time step with Dirac \delta(x-init) initial condition
    G = GMatrix.computeG(xvec, xvec, h)
    return xvec, phat, G

# Figure out better initial max and min for the grid.
def correctInitialGrid(xMin, xMax, a, b, k):
    machEps = np.finfo(float).eps
    tol = 1
    while a <= xMin:
        xMin = np.round(xMin - tol, 3)
        xMax = a + tol

    while a >= xMax:
        xMax = np.round(xMax + tol, 3)
        xMin = a - tol

    xvec = np.arange(xMin, xMax, k)
    phat = fun.dnorm(xvec, a, b)
    numNonzero = np.sum(phat > machEps)
    tol = len(phat) * 0.3
    while (numNonzero < tol) & (k > 0.001):  # check if we need to make the grid size finer.
        k = k * 0.5
        xvec = np.arange(xMin, xMax, k)
        phat = fun.dnorm(xvec, a, b)
        numNonzero = np.sum(phat > machEps)
    return xvec, k, phat