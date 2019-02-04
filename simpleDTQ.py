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

machEps = np.finfo(float).eps


# Drift function
def driftfun(x):
    if isinstance(x, int) | isinstance(x, float):
        return 4
    else:
        return np.ones(np.shape(x)) * 4
    return x * (4 - x ** 2)


# Diffusion function
def difffun(x):
    return np.repeat(0.1, np.size(x))


# Density, distribution function, quantile function and random generation for the
# normal distribution with mean equal to mu and standard deviation equal to sigma.
def dnorm(x, mu, sigma):
    return np.divide(1, (sigma * np.sqrt(2 * np.pi))) * np.exp(np.divide(-(x - mu) ** 2, 2 * sigma ** 2))


# visualization parameters ############################################################
finalGraph = False
animate = True
plotEvolution = False
plotEps = False
animateIntegrand = True
plotGSizeEvolution = True
plotLargestEigenvector = False
plotIC = False

# tolerance parameters
epsilonTolerance = -20
minSizeGAndStillRemoveValsFromG = 60
minMaxOfPhatAndStillRemoveValsFromG = 0.01

# simulation parameters
autoCorrectInitialGrid = True
RemoveFromG = True
AddToG = True
T = 1  # final time, code computes PDF of X_T
s = 0.75  # the exponent in the relation k = h^s
h = 0.01  # temporal step size
init = 0  # initial condition X_0
numsteps = int(np.ceil(T / h))

assert numsteps > 0, 'The variable numsteps must be greater than 0'

# define spatial grid
k = h ** s
xMin = -1
xMax = 1
################################################################################

# pdf after one time step with Dirac delta(x-init) initial condition
a = init + driftfun(init)
b = np.abs(difffun(init)) * np.sqrt(h)

if not autoCorrectInitialGrid:
    xvec = np.arange(xMin, xMax, k)
    phat = dnorm(xvec, a, b)

else:
    xvec, k, phat = XGrid.correctInitialGrid(xMin, xMax, a, b, k, dnorm)

# Kernel matrix
G = GMatrix.computeG(xvec, xvec, h, driftfun, difffun, dnorm)

if plotIC:
    plt.figure()
    plt.plot(xvec, phat)
    plt.show()

pdf_trajectory = []
xvec_trajectory = []
epsilonArray = []
G_history = []
epsilonArray.append(Integrand.computeEpsilon(G, phat))

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

        ############################################## removing from grid
        changedG = False
        if RemoveFromG & (len(G) > minSizeGAndStillRemoveValsFromG) & (countSteps > 10):
            valsToRemove = GMatrix.checkReduceG(G, pdf)  # Remove if val is -inf
            if -np.inf in valsToRemove:
                for ind_w in reversed(range(len(valsToRemove))):  # reversed to avoid index problems
                    if (valsToRemove[ind_w] == -np.inf) & (
                            np.max(pdf > minMaxOfPhatAndStillRemoveValsFromG)) & (
                            len(G) > minSizeGAndStillRemoveValsFromG):
                        G = GMatrix.removeGridValuesFromG(ind_w, G)
                        changedG = True
                        xvec = np.delete(xvec, ind_w)
                        pdf = np.delete(pdf, ind_w)
            ############################################################
        if changedG:
            epsilon = Integrand.computeEpsilon(G, pdf)

        if epsilon > epsilonTolerance:
            IC = False
            if len(xvec_trajectory) < 2:  # pdf trajectory size is 1
                IC = True
            while AddToG & (epsilon >= epsilonTolerance):
                ############################################## adding to grid
                leftEnd = xvec[0] - k
                rightEnd = xvec[-1] + k
                G = GMatrix.addGridValueToG(xvec, leftEnd, h, driftfun, difffun, G, 0, dnorm)
                xLoc, xvec = XGrid.addValueToXvec(xvec, leftEnd)
                pdf = np.insert(pdf, xLoc, 0)
                G = GMatrix.addGridValueToG(xvec, rightEnd, h, driftfun, difffun, G, len(G), dnorm)
                xLoc, xvec = XGrid.addValueToXvec(xvec, rightEnd)
                pdf = np.insert(pdf, xLoc, 0)
                epsilon = Integrand.computeEpsilon(G, G * pdf)
                epsilonArray.append(epsilon)
                print(epsilon)
                ################################################
            # recompute ICs with new xvec. "restart"
            if IC:
                pdf_trajectory[-1] = dnorm(xvec, init + driftfun(init),
                                           np.abs(difffun(init)) * np.sqrt(h))

        if epsilon <= epsilonTolerance:  # things are going well
            pdf_trajectory.append(np.dot(G * k, pdf))  # Equispaced Trapezoidal Rule
            xvec_trajectory.append(xvec)
            if animateIntegrand | plotGSizeEvolution: G_history.append(G)
            epsilonArray.append(Integrand.computeEpsilon(G, pdf))
            countSteps = countSteps + 1

    # Animate the PDF
    f1 = plt.figure()
    l = f1.add_subplot(1, 1, 1)
    im, = l.plot([], [], 'r')
    NeedToChangeXAxes, NeedToChangeYAxes, starting_minxgrid, starting_maxxgrid, starting_maxygrid = AnimationTools.axis_setup(
        xvec_trajectory, pdf_trajectory)
    anim = animation.FuncAnimation(f1, AnimationTools.update_animation, len(xvec_trajectory),
                                   fargs=(pdf_trajectory, l, xvec_trajectory, im, NeedToChangeXAxes, NeedToChangeYAxes,
                                          starting_minxgrid, starting_maxxgrid, starting_maxygrid), interval=50,
                                   blit=False)
    plt.show()

if animateIntegrand:
    assert animate == True, 'Animate must be True'
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
    vals, vects = np.linalg.eig(G_history[-1])
    largest_eigenvector = GMatrix.scaleEigenvector(vects[:, 0], k * np.ones(len(vects[:, 0])))
    plt.figure()
    plt.plot(xvec_trajectory[-1], pdf_trajectory[-1], label='PDF')
    plt.plot(xvec_trajectory[-1], np.real(largest_eigenvector), '.k', label='Eigenvector')  # blue
    plt.legend()
    plt.show()
