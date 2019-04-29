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
finalGraph = True
animate = True
plotEvolution = False
plotEps = False
animateIntegrand = False
plotGSizeEvolution = False
plotLargestEigenvector_equispaced = False
plotIC = False

# tolerance parameters
epsilonTolerance = -30
minSizeGAndStillRemoveValsFromG = 100
minMaxOfPhatAndStillRemoveValsFromG = 0.001

# simulation parameters
autoCorrectInitialGrid = True
RandomXvec = False  # if autoCorrectInitialGrid is True this has no effect.

RemoveFromG = True  # Also want AddToG to be true if true
IncGridDensity = True
DecGridDensity = True
AddToG = True

T = 1.  # final time, code computes PDF of X_T
s = 0.75  # the exponent in the relation k = h^s
h = 0.01  # temporal step size
init = 0  # initial condition X_0
numsteps = int(np.ceil(T / h))

assert numsteps > 0, 'The variable numsteps must be greater than 0'

# define spatial grid
k = h ** s
xMin = -2
xMax = 2
################################################################################

pdf_trajectory = []
xvec_trajectory = []
epsilonArray = []
G_history = []
steepnessArr = []
kvec_trajectory = []
diff = []

if fun.difffun(np.random) == 0:
    xvec = np.arange(xMin, xMax, k)
    countSteps = 0
    phat0 = lambda xx: 1 * (1 / k) * ((xx > - k) & (xx < k))
    pdf_trajectory.append(np.copy(phat0(xvec)))
    xvec_trajectory.append(np.copy(xvec))
    while countSteps < numsteps - 1:
        xvec = xvec+fun.driftfun(xvec) * h
        xvec_trajectory.append(np.copy(xvec))
        pdf = phat0(xvec_trajectory[0])
        pdf_trajectory.append(np.copy(pdf))
        countSteps += 1

else:
    a = init + fun.driftfun(init)
    b = np.abs(fun.difffun(init)) * np.sqrt(h)

    if not autoCorrectInitialGrid:
        if not RandomXvec: xvec = np.arange(xMin, xMax, k)
        if RandomXvec:
            xvec = XGrid.getRandomXgrid(xMin, xMax, 2000)
        xvec = XGrid.densifyGridAroundDirac(xvec, a, k)
        # plt.figure()
        # plt.plot(xvec, '.', markersize=0.5)
        # plt.show()
        phat = fun.dnorm(xvec, a, b)  # pdf after one time step with Dirac \delta(x-init) initial condition

        G = GMatrix.computeG(xvec, xvec, h)

    else:
        xvec = np.arange(xMin, xMax, k)
        phat = fun.dnorm(xvec, a, b)
        xvec, k, phat = XGrid.correctInitialGrid(xMin, xMax, a, b, k)
        G = GMatrix.computeG(xvec, xvec, h)
        # if IncGridDensity: xvec, G, phat, gradVal = XGrid.addPointsToGridBasedOnGradient(xvec, phat, h, G)

    # xvec = pickle.load(open("xvec.p", "rb"))
    # t = np.min(np.diff(xvec))
    # # Kernel matrix
    # G = GMatrix.computeG(xvec, xvec, h)
    # phat = fun.dnorm(xvec, a, b)  # pdf after one time step with Dirac delta(x-init) initial condition

    if plotIC:
        plt.figure()
        plt.plot(xvec, phat)
        plt.show()

    epsilonArray.append(Integrand.computeEpsilon(G, phat))
    pdf_trajectory.append(phat)  # solution after one time step from above
    xvec_trajectory.append(xvec)
    G_history.append(G)

    countSteps = 0
    while countSteps < numsteps - 1:  # since one time step is computed above
        print(countSteps)
        pdf = pdf_trajectory[-1]  # set up placeholder variables
        xvec = xvec_trajectory[-1]
        if countSteps > 0:  # Editing grid interior
            xvec, pdf, G = XGrid.adjustGrid(xvec, pdf, G, k, h, xvec_trajectory[-2], pdf_trajectory[-2],
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
                leftEnd = xvec[0] - (xvec[1] - xvec[0])
                rightEnd = xvec[-1] + (xvec[-1] - xvec[-2])
                G = GMatrix.addGridValueToG(xvec, leftEnd, h, G, 0)
                xLoc, xvec = XGrid.addValueToXvec(xvec, leftEnd)
                pdf = np.insert(pdf, xLoc, 0)
                G = GMatrix.addGridValueToG(xvec, rightEnd, h, G, len(G))
                xLoc, xvec = XGrid.addValueToXvec(xvec, rightEnd)
                pdf = np.insert(pdf, xLoc, 0)
                epsilon = Integrand.computeEpsilon(G, G * pdf)
                epsilonArray.append(epsilon)
                # plt.figure()
                # plt.plot(xvec,pdf, '.')
                # plt.show()
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
            G_history.append(G)
            epsilonArray.append(Integrand.computeEpsilon(G, pdf))
            countSteps = countSteps + 1
            t = 0

    # Animate the PDF
if animate:
    f1 = plt.figure()
    l = f1.add_subplot(1, 1, 1)
    im, = l.plot([], [], 'r', markersize=1)
    NeedToChangeXAxes, NeedToChangeYAxes, starting_minxgrid, starting_maxxgrid, starting_maxygrid = AnimationTools.axis_setup(
        xvec_trajectory, pdf_trajectory)
    anim = animation.FuncAnimation(f1, AnimationTools.update_animation, len(xvec_trajectory),
                                   fargs=(
                                       pdf_trajectory, l, xvec_trajectory, im, NeedToChangeXAxes, NeedToChangeYAxes,
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
                                          NeedToChangeYAxes, starting_minxgrid, starting_maxxgrid,
                                          starting_maxygrid),
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

if plotLargestEigenvector_equispaced:
    Gk = k * G
    vect = GMatrix.computeEigenvector(Gk)
    scaled = GMatrix.scaleEigenvector_equispaced(vect, k)
    plt.figure()
    plt.plot(xvec, scaled)
    plt.plot(xvec, pdf)
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

# randSoln = pickle.load(open("trueSoln.p", "rb"))
# randX = pickle.load(open("trueSolnX.p", "rb"))
#
# plt.figure()
# plt.plot(randX, randSoln)
# plt.plot(xvec_trajectory[-1], pdf_trajectory[-1])
# plt.show()
