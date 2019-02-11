import numpy as np
import GMatrix
import matplotlib.pyplot as plt



# Adds the new value to the xvec grid in the correct location based on numerical order
def addValueToXvec(xvec, newVal):
    xnewLoc = np.searchsorted(xvec, newVal)
    xvec_new = np.insert(xvec, xnewLoc, newVal)
    return xnewLoc, xvec_new


# Figure out better initial max and min for the grid.
def correctInitialGrid(xMin, xMax, a, b, k, dnorm):
    machEps = np.finfo(float).eps
    tol = 1
    while a <= xMin:
        xMin = np.round(xMin - tol, 3)
        xMax = a + tol

    while a >= xMax:
        xMax = np.round(xMax + tol, 3)
        xMin = a - tol

    xvec = np.arange(xMin, xMax, k)
    phat = dnorm(xvec, a, b)
    numNonzero = np.sum(phat > machEps)
    tol = len(phat) * 0.3
    while (numNonzero < tol) & (k > 0.001):  # check if we need to make the grid size finer.
        k = k * 0.5
        xvec = np.arange(xMin, xMax, k)
        phat = dnorm(xvec, a, b)
        numNonzero = np.sum(phat > machEps)

    return xvec, k, phat


def getKvect(xvec):
    kvec = []
    for i in range(1, len(xvec)):
        kvec.append(xvec[i] - xvec[i - 1])
    return np.asarray(kvec)


def getRandomXgrid(beg, end, numVals):
    xvec = (end - beg) * np.random.rand(numVals) + beg
    xvec.sort()
    return xvec


def addPointsToGridBasedOnGradient(xvec, pdf, h, driftfun, difffun, G, dnorm):
    gradVect = np.abs(np.gradient(pdf, xvec))
    xOrig = xvec
    valsAdded = 0
    for i in (range(1, len(xOrig))):  # all points except last one
        if gradVect[i] > 2:
            curr = xOrig[i]
            left = xOrig[i - 1]
            grad = np.ceil(gradVect[i])
            if (curr-left > 0.0001):
                grad = min(grad+1, 4)
                valsToAdd = []
                for count in range(int(grad)-1):
                    val = curr - np.abs((count + 1) * ((curr - left) / grad))
                    if (val not in xvec):
                        valsToAdd.append(val)
                        valsAdded += 1
                for add in valsToAdd:
                    xnewLoc, xvecNew = addValueToXvec(xvec, add)
                    G = GMatrix.addGridValueToG(xvec, add, h, driftfun, difffun, G, xnewLoc, dnorm)
                    mid = np.round((pdf[xnewLoc - 1] + pdf[xnewLoc]) / 2)
                    pdf = np.insert(pdf, xnewLoc, mid)
                    xvec = xvecNew

    # plt.figure()
    # plt.plot(xvec)
    # plt.show()
    return xvec, G, pdf


def removePointsFromGridBasedOnGradient(xvec, pdf, k, G):
    gradVect = np.abs(np.gradient(pdf, xvec))
    if len(G)>100:
        for i in reversed(range(len(gradVect)-2)):  # all points except last one
            curr = xvec[i]
            right2 = xvec[i + 2]
            q = (curr - right2) / 2
            if (gradVect[i] < 0.1) & ((right2-curr) < k) & (len(G)>100):
                G = GMatrix.removeGridValuesFromG(i+1, G)
                xvec = np.delete(xvec, i+1)
                pdf = np.delete(pdf, i+1)
    return xvec, G, pdf
