# code to compute the PDF of the solution of the SDE:
#
# dX_t = X_t*(4-X_t^2) dt + dW_t
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import Integrand
import AnimationTools
import GMatrix
import XGrid
import QuadRules
import Functions as fun
import pickle

machEps = np.finfo(float).eps

# visualization parameters ############################################################
finalGraph = False
animate = True
plotEvolution = False
plotEps = False
animateIntegrand = True
plotGSizeEvolution = True
plotLargestEigenvector = True
plotIC = False

# tolerance parameters
epsilonTolerance = -30
minSizeGAndStillRemoveValsFromG = 100
minMaxOfPhatAndStillRemoveValsFromG = 0.0001

# simulation parameters
autoCorrectInitialGrid = False
RandomXvec = False  # if autoCorrectInitialGrid is True this has no effect.

RemoveFromG = True  # Also want AddToG to be true if true
IncGridDensity = True
DecGridDensity = True
AddToG = True

T = 1  # final time, code computes PDF of X_T
s = 0.75  # the exponent in the relation k = h^s
h = 0.01  # temporal step size
init = 0  # initial condition X_0
numsteps = int(np.ceil(T / h))

assert numsteps > 0, 'The variable numsteps must be greater than 0'

# define spatial grid
k = h ** s
xMin = -5
xMax = 5
################################################################################

a = init + fun.driftfun(init)
b = np.abs(fun.difffun(init)) * np.sqrt(h)

if not autoCorrectInitialGrid:
    if not RandomXvec: xvec = np.arange(xMin, xMax, k)
    if RandomXvec:
        xvec = XGrid.getRandomXgrid(xMin, xMax, 2000)
    xvec = XGrid.densifyGridAroundDirac(xvec, a, k)
    plt.figure()
    plt.plot(xvec, '.', markersize=0.5)
    plt.show()
    phat = fun.dnorm(xvec, a, b) # pdf after one time step with Dirac \delta(x-init) initial condition

    G = GMatrix.computeG(xvec, xvec, h)

else:
    xvec = np.arange(xMin, xMax, k)
    phat = fun.dnorm(xvec, a, b)
    xvec, k, phat = XGrid.correctInitialGrid(xMin, xMax, a, b, k)
    G = GMatrix.computeG(xvec, xvec, h)
    #if IncGridDensity: xvec, G, phat, gradVal = XGrid.addPointsToGridBasedOnGradient(xvec, phat, h, G)

# xvec = pickle.load(open("xvec.p", "rb"))
# t = np.min(np.diff(xvec))
# # Kernel matrix
# G = GMatrix.computeG(xvec, xvec, h)
# phat = fun.dnorm(xvec, a, b)  # pdf after one time step with Dirac delta(x-init) initial condition


if plotIC:
    plt.figure()
    plt.plot(xvec, phat)
    plt.show()

pdf_trajectory = []
xvec_trajectory = []
epsilonArray = []
G_history = []
epsilonArray.append(Integrand.computeEpsilon(G, phat))
steepnessArr = []
kvec_trajectory = []
diff = []

if animate:
    pdf_trajectory.append(phat)  # solution after one time step from above
    xvec_trajectory.append(xvec)
    if animateIntegrand | plotGSizeEvolution: G_history.append(G)

    countSteps = 0
    while countSteps < numsteps - 1:  # since one time step is computed above
        print(countSteps)
        pdf = pdf_trajectory[-1]  # set up placeholder variables
        xvec = xvec_trajectory[-1]
        epsilon = Integrand.computeEpsilon(G, pdf)
        ############################################ Densify grid
        if (countSteps > 0) & IncGridDensity:
            steepness = np.gradient(pdf, xvec)
            steepnessArr.append(abs(steepness))
            x = len(xvec)
            xvec, G, pdf, gradVal = XGrid.addPointsToGridBasedOnGradient(xvec, pdf, h, G, pdf_trajectory[-2], xvec_trajectory[-2], G_history[-2])
            diff.append(len(xvec) - x)
        # ############################################
        if DecGridDensity & (countSteps >10):
            xvec, G, pdf = XGrid.removePointsFromGridBasedOnGradient(xvec, pdf, k, G,h)
        ############################################# removing from grid
        if RemoveFromG & (len(G) > minSizeGAndStillRemoveValsFromG) & (countSteps > 10):
            valsToRemove = GMatrix.checkReduceG(G, pdf)  # Remove if val is -inf
            if -np.inf in valsToRemove:
                for ind_w in reversed(range(len(valsToRemove))):  # reversed to avoid index problems
                    if (valsToRemove[ind_w] == -np.inf) & (
                            np.max(pdf > minMaxOfPhatAndStillRemoveValsFromG)) & (
                            len(G) > minSizeGAndStillRemoveValsFromG):
                        G = GMatrix.removeGridValuesFromG(ind_w, G)
                        xvec = np.delete(xvec, ind_w)
                        pdf = np.delete(pdf, ind_w)
            ############################################################
        epsilon = Integrand.computeEpsilon(G, pdf)
        if epsilon > epsilonTolerance:
            IC = False
            if len(xvec_trajectory) < 2:  # pdf trajectory size is 1
                IC = True
            while AddToG & (epsilon >= epsilonTolerance):
                ############################################## adding to grid
                leftEnd = xvec[0] - k
                rightEnd = xvec[-1] + k
                G = GMatrix.addGridValueToG(xvec, leftEnd, h, G, 0)
                xLoc, xvec = XGrid.addValueToXvec(xvec, leftEnd)
                pdf = np.insert(pdf, xLoc, 0)
                G = GMatrix.addGridValueToG(xvec, rightEnd, h, G, len(G))
                xLoc, xvec = XGrid.addValueToXvec(xvec, rightEnd)
                pdf = np.insert(pdf, xLoc, 0)
                epsilon = Integrand.computeEpsilon(G, G * pdf)
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
            kvect = XGrid.getKvect(xvec)
            kvec_trajectory.append(kvect)
            pdf_trajectory.append(QuadRules.TrapUnequal(G, pdf, kvect))  # nonequal Trap rule
            xvec_trajectory.append(xvec)
            if animateIntegrand | plotGSizeEvolution: G_history.append(G)
            epsilonArray.append(Integrand.computeEpsilon(G, pdf))
            countSteps = countSteps + 1
            t = 0

    # Animate the PDF
    f1 = plt.figure()
    l = f1.add_subplot(1, 1, 1)
    im, = l.plot([], [], '.r', markersize=3)
    NeedToChangeXAxes, NeedToChangeYAxes, starting_minxgrid, starting_maxxgrid, starting_maxygrid = AnimationTools.axis_setup(
        xvec_trajectory, pdf_trajectory)
    anim = animation.FuncAnimation(f1, AnimationTools.update_animation, len(xvec_trajectory),
                                   fargs=(pdf_trajectory, l, xvec_trajectory, im, NeedToChangeXAxes, NeedToChangeYAxes,
                                          starting_minxgrid, starting_maxxgrid, starting_maxygrid), interval=50,
                                   blit=False)
    plt.show()

if animateIntegrand:
    assert animate, 'Animate must be True'
    f1 = plt.figure()
    l = f1.add_subplot(1, 1, 1)
    im, = l.plot([], [], 'r')
    NeedToChangeXAxes, NeedToChangeYAxes, starting_minxgrid, starting_maxxgrid, starting_maxygrid = AnimationTools.axis_setup(
        xvec_trajectory, pdf_trajectory)
    anim = animation.FuncAnimation(f1, AnimationTools.update_animation_integrand, len(xvec_trajectory),
                                   fargs=(G_history, l, xvec_trajectory, pdf_trajectory, im, NeedToChangeXAxes,
                                          NeedToChangeYAxes, starting_minxgrid, starting_maxxgrid, starting_maxygrid),
                                   interval=50,
                                   blit=False)
    plt.show()

if plotEps:
    assert animate == True, 'The variable animate must be True'
    plt.figure()
    plt.plot(epsilonArray)
    plt.xlabel('Time Step')
    plt.ylabel(r'$\varepsilon$ value')
    plt.title(
        r'$\varepsilon$ at each time step for $f(x)=x(4-x^2), g(x)=1, k \approx 0.032$, tol = -20')
    plt.show()

if plotEvolution:
    assert animate == True, 'The variable animate must be True'
    plt.figure()
    plt.suptitle(r'Evolution for $f(x)=x(4-x^2), g(x)=1, k \approx 0.032$')
    numPDF = len(pdf_trajectory)
    plt.subplot(2, 2, 1)
    plt.title("t=0")
    plt.plot(xvec_trajectory[0], pdf_trajectory[0])
    plt.subplot(2, 2, 2)
    plt.title("t=T/3")
    plt.plot(xvec_trajectory[int(np.ceil(numPDF * (1 / 3)))], pdf_trajectory[int(np.ceil(numPDF * (1 / 3)))])
    plt.subplot(2, 2, 3)
    plt.title("t=2T/3")
    plt.plot(xvec_trajectory[int(np.ceil(numPDF * (2 / 3)))], pdf_trajectory[int(np.ceil(numPDF * (2 / 3)))])
    plt.subplot(2, 2, 4)
    plt.title("t=T")
    plt.plot(xvec_trajectory[int(np.ceil(numPDF - 1))], pdf_trajectory[int(np.ceil(numPDF - 1))])
    plt.show()

if finalGraph:
    assert animate == True, 'The variable animate must be True'
    plt.plot(xvec_trajectory[-1], pdf_trajectory[-1], '.')
    plt.show()

if plotGSizeEvolution:
    plt.figure()
    for j, i in enumerate(G_history):
        w = i.shape[0]
        plt.plot(j, w, '.')
    plt.title(
        r'Size of G at each time step for $f(x)=x(4-x^2), g(x)=1, k \approx 0.032$, starting interval [-1,1], tol = -100')
    plt.show()


if plotLargestEigenvector:
    index = -1
    kvect = XGrid.getKvect(xvec)
    kvect = np.insert(kvect,50,0.1)
    vals, vects = np.linalg.eig(G_history[-1])
    vals = np.real(vals)
    largest_eigenvector_unscaled = vects[:, 0]
    #largest_eigenvector = GMatrix.scaleEigenvector(vects[:,1], kvect * np.ones(len(vects[:, 0])))
    #largest_eigenvector1 = GMatrix.scaleEigenvector(vects[:,2], kvect * np.ones(len(vects[:, 0])))
    w = np.real(vals)
    plt.figure()
    plt.plot(xvec_trajectory[-1], pdf_trajectory[-1], label='PDF')
    plt.plot(xvec_trajectory[-1],np.real(largest_eigenvector_unscaled), '.k', label='Eigenvector')  # blue
    #plt.plot(xvec_trajectory[-1],np.real(largest_eigenvector1), '.k', label='Eigenvector')  # blue
    plt.legend()
    plt.show()

plotXVecEvolution = True
if plotXVecEvolution:
    plt.figure()
    for i in range(len(xvec_trajectory)):
        plt.plot(xvec_trajectory[i], np.ones(len(xvec_trajectory[i])) * i, '.', markersize=0.1)
        # plt.plot(kvec_trajectory[i])
        # plt.plot(xvec_trajectory[i])

    plt.show()

# file = open('trueSoln.p', 'wb')
# pickle.dump(pdf_trajectory[-1], file)
# file.close()
# file = open('trueSolnX.p', 'wb')
# pickle.dump(xvec_trajectory[-1], file)
# file.close()

# plt.figure()
# plt.plot(xvec_trajectory[0],pdf_trajectory[0], '.')
# plt.show()
# Integrand.plotIntegrand(G_history[1],pdf_trajectory[1],xvec_trajectory[1])

randSoln = pickle.load(open("trueSoln.p", "rb"))
randX = pickle.load(open("trueSolnX.p", "rb"))

plt.figure()
plt.plot(randX,randSoln)
plt.plot(xvec_trajectory[-1], pdf_trajectory[-1])
plt.show()
