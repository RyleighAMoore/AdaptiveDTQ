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


def densifyGridAroundDirac(xvec, center_a, k):
    left = center_a - 0.5
    right = center_a + 0.5
    div = 16
    denser = np.arange(left + k / div, right - k / div, k / div)
    xvec = np.concatenate((xvec, denser), axis=0)
    xvec.sort(axis=0)
    return xvec


def adjustGrid(xvec, pdf, G, k, h, xvecPrev, pdfPrev, GPrev, countSteps):
    # plt.figure()
    # plt.plot(xvec,pdf, '.')
    # plt.show()
    pdfPrev2 = np.copy(pdf)
    xvecPrev2 = np.copy(xvec)
    Gx = GMatrix.computeG_partialx(xvec, xvec, h)
    kvect = getKvect(xvec)
    gradVect = abs(QuadRules.TrapUnequal(Gx, pdf, kvect))
    xvecOrig = xvec
    gradTol = 1
    maxPdfTol = 0.000001
    GMinTol = 0
    removeArr = GMatrix.checkReduceG(G, pdf, 10**(-5))
    for x in xvecOrig[1:-1]:
        if x in xvec:  # Check that hasn't been removed
            xvecLoc = np.where(xvec == x)
            t = np.size(xvecLoc)
            assert t < 2, 'Returned same value in list twice'
            xvecLoc = xvecLoc[0][0]
            xvecOrigLoc = np.where(xvecOrig == x)
            assert np.size(xvecLoc) < 2, 'Returned same value in list twice'
            xvecOrigLoc = xvecOrigLoc[0][0]
            if np.size(xvecLoc) == 1:  # If value is in list once
                ################# Add values left and right if gradient is large enough
                if gradVect[xvecOrigLoc] > gradTol:
                    # xvecLoc = np.where(xvec == x)
                    # t = np.size(xvecLoc)
                    # assert t < 2, 'Returned same value in list twice'
                    # xvecLoc = xvecLoc[0][0]
                    # xvecOrigLoc = np.where(xvecOrig == x)
                    # assert np.size(xvecLoc) < 2, 'Returned same value in list twice'
                    # xvecOrigLoc = xvecOrigLoc[0][0]
                    left = xvecOrig[xvecOrigLoc - 1]
                    right = xvecOrig[xvecOrigLoc + 1]
                    valLeft = x - (x - left) / 2
                    valRight = x + (right - x) / 2
                    if (valRight not in xvec) & ((xvec[xvecLoc + 1] - x) > k / 4):
                        ################### Add right val
                        G = GMatrix.addGridValueToG(xvec, valRight, h, G, xvecLoc + 1)
                        pdf = GMatrix.getNewPhatWithNewValue(xvecPrev, valRight, h, pdf, pdfPrev, xvecLoc + 1, GPrev)
                        temp, xvecNew = addValueToXvec(xvec, valRight)
                        xvec = xvecNew
                        # plt.figure()
                        # plt.plot(xvec,pdf,'.r')
                        # plt.show()
                        t = 0
                        if (valLeft not in xvec) & ((x - xvec[xvecLoc - 1]) > k / 4):
                            ################## Add left val
                            G = GMatrix.addGridValueToG(xvec, valLeft, h, G, xvecLoc)
                            pdf = GMatrix.getNewPhatWithNewValue(xvecPrev, valLeft, h, pdf, pdfPrev, xvecLoc, GPrev)
                            temp, xvecNew = addValueToXvec(xvec, valLeft)
                            xvec = xvecNew
                            # plt.figure()
                            # plt.plot(xvec,pdf,'.b')
                            # plt.show()
                ####################################################
                ######################### Remove b/c value is zero
                    xvecLoc = np.where(xvec == x)
                    t = np.size(xvecLoc)
                    assert t < 2, 'Returned same value in list twice'
                    xvecLoc = xvecLoc[0][0]
                    xvecOrigLoc = np.where(xvecOrig == x)
                    assert np.size(xvecLoc) < 2, 'Returned same value in list twice'
                    xvecOrigLoc = xvecOrigLoc[0][0]
                elif ((removeArr[xvecOrigLoc-1]==-np.inf )| (removeArr[xvecOrigLoc+1]==-np.inf))&(countSteps > 1) & (removeArr[xvecOrigLoc] == -np.inf) & (np.max(pdf) > maxPdfTol) & (
                        len(G) > GMinTol):
                    print('zeros')
                    if xvecLoc == np.size(xvec) - 2:  # If second from end
                        G = GMatrix.removeGridValueIndexFromG(np.size(xvec) - 1, G)
                        xvec = np.delete(xvec, np.size(xvec) - 1)
                        pdf = np.delete(pdf, np.size(xvec) - 1)
                    G = GMatrix.removeGridValueIndexFromG(xvecLoc, G)
                    xvec = np.delete(xvec, xvecLoc)
                    pdf = np.delete(pdf, xvecLoc)
                    if xvecLoc == 1:  # If second from start
                        G = GMatrix.removeGridValueIndexFromG(0, G)
                        xvec = np.delete(xvec, 0)
                        pdf = np.delete(pdf, 0)
                ###############################################
                ############ #remove values due to gradient
                elif (countSteps > 5) & (gradVect[xvecOrigLoc] <= 0.01) & (
                        xvec[xvecLoc + 1] - xvec[xvecLoc - 1] <= k):
                    G = GMatrix.removeGridValueIndexFromG(xvecLoc, G)
                    xvec = np.delete(xvec, xvecLoc)
                    pdf = np.delete(pdf, xvecLoc)
                    print('removed')
                    t = 0
                ####################################################

    return xvec, pdf, G
