import numpy as np
import GMatrix
import matplotlib.pyplot as plt
import QuadRules
import Functions as fun


# Adds the new value to the xvec grid in the correct location based on numerical order
def addValueToXvec(xvec, newVal):
    xnewLoc = np.searchsorted(xvec, newVal)
    xvec_new = np.insert(xvec, xnewLoc, newVal)
    return xnewLoc, xvec_new


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


def getKvect(xvec):
    kvec = []
    for i in range(1, len(xvec)):
        kvec.append(xvec[i] - xvec[i - 1])
    return np.asarray(kvec)


def getRandomXgrid(beg, end, numVals):
    xvec = (end - beg) * np.random.rand(numVals) + beg
    xvec.sort()
    return xvec


def addPointsToGridBasedOnGradient(xvec, pdf, h, G, pdfOld, xvecOld, Gold):
    Gx = GMatrix.computeG_partialx(xvec, xvec, h)
    kvect = getKvect(xvec)
    gradVect = abs(QuadRules.TrapUnequal(Gx, pdf, kvect))
    # plt.figure()
    # plt.plot(xvec, pdf, '.')
    xOrig = xvec
    tol = 10
    addedLeftAlready = False
    i = 1
    while i < len(xOrig):  # all points except first one
        if gradVect[i] > 5:
            curr = xOrig[i]
            left = xOrig[i - 1]
            right = xOrig[i + 1]
            if (curr - left > h) & (not addedLeftAlready): # Add left value
                valLeft = curr - (curr - left) / 2
                xnewLoc, xvecNew = addValueToXvec(xvec, valLeft)
                xoldLoc, temp = addValueToXvec(xvecOld, valLeft)
                G = GMatrix.addGridValueToG(xvec, valLeft, h, G, xnewLoc)
                mid = np.round((pdf[xnewLoc - 1] + pdf[xnewLoc]) / 2)
                pdf = GMatrix.getNewPhatWithNewValue(xvecOld, valLeft, h, pdf, pdfOld, xnewLoc, Gold, xoldLoc)
                #pdf = np.insert(pdf, xnewLoc, mid)
                xvec = xvecNew

            if right - curr > h:  # Add right value
                valRight = curr + (right - curr) / 2
                addedLeftAlready = True
                xnewLoc, xvecNew = addValueToXvec(xvec, valRight)
                xoldLoc, temp = addValueToXvec(xvecOld, valRight)
                G = GMatrix.addGridValueToG(xvec, valRight, h, G, xnewLoc)
                mid = np.round((pdf[xnewLoc - 1] + pdf[xnewLoc]) / 2)
                pdf = GMatrix.getNewPhatWithNewValue(xvecOld, valRight, h, pdf, pdfOld, xnewLoc, Gold, xoldLoc)
                #pdf = np.insert(pdf, xnewLoc, mid)
                xvec = xvecNew
                i=i+1 # go one further because the
            elif right - curr <= h:
                addedLeftAlready = False
        i = i+1 # next step
    # plt.plot(xvec, pdf, '.')
    # plt.show()
    return xvec, G, pdf, gradVect


def removePointsFromGridBasedOnGradient(xvec, pdf, k, G, h):
    Gx = GMatrix.computeG_partialx(xvec, xvec, h)
    kvect = getKvect(xvec)
    gradVect = abs(QuadRules.TrapUnequal(Gx, pdf, kvect))
    xOrig = xvec
    if len(G) > 200:
        for i in reversed(range(2, len(xOrig)-2)):  # all points except last one
            curr = xOrig[i]
            right2 = xOrig[i + 2]
            left2 = xOrig[i - 2]
            if (gradVect[i] < 10**(-15)) & ((right2 - curr) / 2 < k) & ((curr - left2) / 2 < k) & (len(G) > 200):
                xnewLoc = np.searchsorted(xvec, xOrig[i])
                G = GMatrix.removeGridValuesFromG(xnewLoc, G)
                xvec = np.delete(xvec, xnewLoc)
                pdf = np.delete(pdf, xnewLoc)
                print('Removed')
    return xvec, G, pdf


def densifyGridAroundDirac(xvec, center_a, k):
    left = center_a - 1
    right = center_a + 1
    denser = np.arange(left+k/4, right-k/4, k/4)
    xvec = np.concatenate((xvec,denser),axis=0)
    xvec.sort(axis=0)
    return xvec

