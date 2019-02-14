import numpy as np
import Integrand
import Functions as fun


def scaleEigenvector(eigenvector, stepSizes):
    scale = np.real(np.matmul(eigenvector, stepSizes))
    scale = 1 / scale
    return (scale) * eigenvector


#  Function that returns the kernel matrix G(x,y)
def computeG(xvec, yvec, h):
    Y = np.zeros((len(yvec), len(yvec)))
    for i in range(len(yvec)):
        Y[i, :] = xvec  # Y has the same grid value along each column (col1 has x1, col2 has x2, etc)
    mu = Y + fun.driftfun(Y) * h
    r = fun.difffun(Y)
    sigma = abs(fun.difffun(Y)) * np.sqrt(h)
    sigma = np.reshape(sigma, [np.size(xvec), np.size(yvec)])  # make a matrix for the dnorm function
    Y = np.transpose(Y)  # Transpose Y for use in the dnorm function
    test = fun.dnorm(Y, mu, sigma)
    return test


def computeG_partialx(xvec, yvec, h):
    Y = np.zeros((len(yvec), len(yvec)))
    for i in range(len(yvec)):
        Y[i, :] = xvec  # Y has the same grid value along each column (col1 has x1, col2 has x2, etc)
    mu = Y + fun.driftfun(Y) * h
    r = fun.difffun(Y)
    sigma = abs(fun.difffun(Y)) * np.sqrt(h)
    sigma = np.reshape(sigma, [np.size(xvec), np.size(yvec)])  # make a matrix for the dnorm function
    Y = np.transpose(Y)  # Transpose Y for use in the dnorm function
    test = fun.dnorm_partialx(Y, mu, sigma)
    return test



# This adds a N dimensional row to a M by N dimensional G
def addRowToG(xvec, newVal, h, G, rowIndex):
    mu = xvec + fun.driftfun(xvec) * h
    sigma = abs(fun.difffun(xvec)) * np.sqrt(h)
    xrep = np.ones(len(mu)) * newVal
    newRow = fun.dnorm(xrep, mu, sigma)
    Gnew = np.insert(G, rowIndex, newRow, 0)
    return Gnew


# This adds a M dimensional column to a M by N dimensional G
def addColumnToG(xvec, newVal, h, G, colIndex):
    mu = np.ones(len(G)) * (newVal + fun.driftfun(newVal) * h)
    w = np.ones(len(G)) * newVal
    sigma = abs(fun.difffun(w)) * np.sqrt(h)
    xnewLoc = np.searchsorted(xvec, newVal)
    xnew = np.insert(xvec, xnewLoc, newVal)
    newCol = fun.dnorm(xnew, mu, sigma)
    Gnew = np.insert(G, colIndex, newCol, axis=1)
    return Gnew


# This adds a new grid value to G
def addGridValueToG(xvec, newVal, h,  G, rowIndex):
    G = addRowToG(xvec, newVal, h, G, rowIndex)
    G = addColumnToG(xvec, newVal, h, G, rowIndex)
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
