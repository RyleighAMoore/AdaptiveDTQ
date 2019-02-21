import numpy as np
import Integrand
import Functions as fun
import XGrid
import QuadRules
import matplotlib.pyplot as plt

# Takes in the unscaled eigenvector and the vector of step sizes
# to return an eigenvector with area one underneath.
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

# Computes G_x(x,y)
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
# takes in the old xvec, the new grid value to add,
# the temporal time step h, the matrix G, and the rowIndex of the new value
def addRowToG(xvec, newVal, h, G, rowIndex):
    mu = xvec + fun.driftfun(xvec) * h
    sigma = abs(fun.difffun(xvec)) * np.sqrt(h)
    xrep = np.ones(len(mu)) * newVal
    newRow = fun.dnorm(xrep, mu, sigma)
    Gnew = np.insert(G, rowIndex, newRow, 0)
    return Gnew


def getNewPhatWithNewValue(xvecPrev, newVal, h, phat, phatPrev, xnewLoc, Gold):
    xvecPrevLoc = np.searchsorted(xvecPrev, newVal)
    Gm = addRowToG(xvecPrev, newVal, h, Gold, xvecPrevLoc)
    kvect = XGrid.getKvect(xvecPrev)
    pdfnew = (QuadRules.TrapUnequal(Gm, phatPrev, kvect))
    phatNewVal = pdfnew[xvecPrevLoc]
    phat = np.insert(phat, xnewLoc, phatNewVal)
    # temp, xvecNew = XGrid.addValueToXvec(xvecPrev, newVal)
    return phat


# This adds a M dimensional column to a M by N dimensional G
# Takes in the old xvec, the new grid value to add,
# the temporal time step h, the matrix G, and the column index of the new value
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
# Takes in the old xvec, the new grid value to add,
# the temporal time step h, the matrix G, and the row index of the new value
def addGridValueToG(xvec, newVal, h,  G, rowIndex):
    G = addRowToG(xvec, newVal, h, G, rowIndex)
    G = addColumnToG(xvec, newVal, h, G, rowIndex)
    return G


# This removes a new grid value from G
def removeGridValueIndexFromG(xValIndexToRemove, G):
    G = np.delete(G, xValIndexToRemove, 0)
    G = np.delete(G, xValIndexToRemove, 1)
    return G


# Check if we should remove values from G because they are "zero"
def checkReduceG(G, phat):
    machEps = np.finfo(float).eps
    integrandMaxes = Integrand.computeIntegrandArray(G, phat)
    integrandMaxes[(integrandMaxes <= machEps) & (phat <= machEps)] = -np.inf
    return integrandMaxes


