import numpy as np
import GMatrix
import matplotlib.pyplot as plt
import QuadRules
import Functions as fun
import Integrand
from tqdm import tqdm, trange

# Returns x1 that is centered around 0 and includes (-endingVal:stepSize:endingVal)
def getCenteredZeroXvec(stepSize, endingVal):
    num = int(np.ceil(endingVal/(stepSize)))+1
    x = [0]
    for step in range(1, num):
         x.append(0 + stepSize * step)
         x.append(0 - stepSize * step)
    x = np.sort(np.asarray(x))
    return x


def getCenteredXvecAroundPoint(stepSize, unitsLeft, unitsRight, centerVal):
    numLeft = int(np.ceil(unitsLeft/(stepSize)))+1
    numRight = int(np.ceil(unitsRight/(stepSize)))+1
    x = [centerVal]
    for step in range(1, max(numLeft,numRight)):
        if step < numRight:
            x.append(centerVal + stepSize * step)
        if step < numLeft:
            x.append(centerVal - stepSize * step)
    x = np.sort(np.asarray(x))
    return x


# Adds the new value to the xvec grid in the correct location based on numerical order
def addValueToXvec(xvec, newVal):
    xnewLoc = np.searchsorted(xvec, newVal)
    xvec_new = np.insert(xvec, xnewLoc, newVal)
    return xnewLoc, xvec_new



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
                        t = 0
                        if (valLeft not in xvec) & ((x - xvec[xvecLoc - 1]) > k / 4):
                            ################## Add left val
                            G = GMatrix.addGridValueToG(xvec, valLeft, h, G, xvecLoc)
                            pdf = GMatrix.getNewPhatWithNewValue(xvecPrev, valLeft, h, pdf, pdfPrev, xvecLoc, GPrev)
                            temp, xvecNew = addValueToXvec(xvec, valLeft)
                            xvec = xvecNew
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
                    print('Removing Zero Value')
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


def updateGridExteriors(xvec,h, G, pdf):
    leftEnd = xvec[0] - (xvec[1] - xvec[0])
    rightEnd = xvec[-1] + (xvec[-1] - xvec[-2])
    G = GMatrix.addGridValueToG(xvec, leftEnd, h, G, 0)
    xLoc, xvec = addValueToXvec(xvec, leftEnd)
    pdf = np.insert(pdf, xLoc, 0)
    G = GMatrix.addGridValueToG(xvec, rightEnd, h, G, len(G))
    xLoc, xvec = addValueToXvec(xvec, rightEnd)
    pdf = np.insert(pdf, xLoc, 0)
    epsilon = Integrand.computeEpsilon(G, G * pdf)
    return G, pdf,xvec,epsilon


def stepForwardInTime(countSteps, G, AddToG, pdf_trajectory, xvec_trajectory,IncGridDensity, G_history, epsilonTolerance, epsilonArray, init, kvec_trajectory, k, h):
    print(countSteps)
    pdf = pdf_trajectory[-1]  # set up placeholder variables
    xvec = xvec_trajectory[-1]
    if (countSteps == 0) & IncGridDensity:  # Editing grid interior for first timestep
        xvec, pdf, G = adjustGrid(xvec, pdf, G, k, h, xvec_trajectory[-1], pdf_trajectory[-1],
                                        G_history[-1],
                                        countSteps)
    if (countSteps > 0) & IncGridDensity:  # Editing grid interior
        xvec, pdf, G = adjustGrid(xvec, pdf, G, k, h, xvec_trajectory[-2], pdf_trajectory[-2],
                                        G_history[-2],
                                        countSteps)
    epsilon = Integrand.computeEpsilon(G, pdf)
    print(epsilon)
    if epsilon > epsilonTolerance:
        IC = False
        if len(xvec_trajectory) < 2:  # pdf trajectory size is 1
            IC = True
        while AddToG & (epsilon >= epsilonTolerance):
            #############################################  adding to grid exterior
            G, pdf, xvec, epsilon = updateGridExteriors(xvec, h, G, pdf)
            
            epsilonArray.append(epsilon)
            print(epsilon)
            ################################################
        # recompute ICs with new xvec. "restart"
        if IC:
            pdf_trajectory[-1] = fun.dnorm(xvec, init + fun.driftfun(init),
                                           np.abs(fun.difffun(init)) * np.sqrt(h))

    if epsilon <= epsilonTolerance:  # things are going well
        # pdf_trajectory.append(np.dot(G * k, pdf))  # Equispaced Trapezoidal Rule
        # kvect = np.ones(len(pdf) - 1) * k
        kvect = getKvect(xvec)
        kvec_trajectory.append(kvect)
        pdf_trajectory.append(QuadRules.TrapUnequal(G, pdf, kvect))  # nonequal Trap rule
        xvec_trajectory.append(xvec)
        G_history.append(G)
        epsilonArray.append(Integrand.computeEpsilon(G, pdf))
        countSteps = countSteps + 1
        return countSteps, G, pdf_trajectory, xvec_trajectory, G_history, epsilonArray, kvec_trajectory
    
