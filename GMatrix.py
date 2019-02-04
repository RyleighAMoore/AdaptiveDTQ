import numpy as np
import Integrand


def scaleEigenvector(eigenvector, stepSizes):
    scale = np.real(np.matmul(eigenvector, stepSizes))
    scale = 1 / scale
    return (scale) * eigenvector


#  Function that returns the kernel matrix G(x,y)
def computeG(xvec, yvec, h, driftfun, difffun, dnorm):
    Y = np.zeros((len(yvec), len(yvec)))
    for i in range(len(yvec)):
        Y[i, :] = xvec  # Y has the same grid value along each column (col1 has x1, col2 has x2, etc)
    mu = Y + driftfun(Y) * h
    r = difffun(Y)
    sigma = abs(difffun(Y)) * np.sqrt(h)
    sigma = np.reshape(sigma, [np.size(xvec), np.size(yvec)])  # make a matrix for the dnorm function
    Y = np.transpose(Y)  # Transpose Y for use in the dnorm function
    test = dnorm(Y, mu, sigma)
    return test


# This adds a N dimensional row to a M by N dimensional G
def addRowToG(xvec, newVal, h, driftfun, difffun, G, rowIndex, dnorm):
    mu = xvec + driftfun(xvec) * h
    sigma = abs(difffun(xvec)) * np.sqrt(h)
    xrep = np.ones(len(mu)) * newVal
    newRow = dnorm(xrep, mu, sigma)
    Gnew = np.insert(G, rowIndex, newRow, 0)
    return Gnew


# This adds a M dimensional column to a M by N dimensional G
def addColumnToG(xvec, newVal, h, driftfun, difffun, G, colIndex, dnorm):
    mu = np.ones(len(G)) * (newVal + driftfun(newVal) * h)
    w = np.ones(len(G)) * newVal
    sigma = abs(difffun(w)) * np.sqrt(h)
    xnewLoc = np.searchsorted(xvec, newVal)
    xnew = np.insert(xvec, xnewLoc, newVal)
    newCol = dnorm(xnew, mu, sigma)
    Gnew = np.insert(G, colIndex, newCol, axis=1)
    return Gnew


# This adds a new grid value to G
def addGridValueToG(xvec, newVal, h, driftfun, difffun, G, rowIndex, dnorm):
    G = addRowToG(xvec, newVal, h, driftfun, difffun, G, rowIndex, dnorm)
    G = addColumnToG(xvec, newVal, h, driftfun, difffun, G, rowIndex, dnorm)
    return G


# This removes a new grid value from G
def removeGridValuesFromG(xValIndexToRemove, G):
    G = np.delete(G, xValIndexToRemove, 0)
    G = np.delete(G, xValIndexToRemove, 1)
    return G


# Check if we should remove values from G because they are "zero"
def checkReduceG(G, phat):
    machEps = np.finfo(float).eps
    integrandMaxes = Integrand.computeIntegrandArray(G, phat)
    integrandMaxes[(integrandMaxes < machEps) & (phat < machEps)] = -np.inf
    return integrandMaxes
