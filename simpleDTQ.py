# code to compute the PDF of the solution of the SDE:
#
# dX_t = X_t*(4-X_t^2) dt + dW_t
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import Integrand
import AnimationTools

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

#  Function that returns the kernel matrix G(x,y)
def integrandmat(xvec, yvec, h, driftfun, difffun):
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
def addRowToG(xvec, newVal, h, driftfun, difffun, G, rowIndex):
    mu = xvec + driftfun(xvec) * h
    sigma = abs(difffun(xvec)) * np.sqrt(h)
    xrep = np.ones(len(mu)) * newVal
    newRow = dnorm(xrep, mu, sigma)
    Gnew = np.insert(G, rowIndex, newRow, 0)
    return Gnew


# This adds a M dimensional column to a M by N dimensional G
def addColumnToG(xvec, newVal, h, driftfun, difffun, G, colIndex):
    mu = np.ones(len(G)) * (newVal + driftfun(newVal) * h)
    w = np.ones(len(G)) * newVal
    sigma = abs(difffun(w)) * np.sqrt(h)
    xnewLoc = np.searchsorted(xvec, newVal)
    xnew = np.insert(xvec, xnewLoc, newVal)
    newCol = dnorm(xnew, mu, sigma)
    Gnew = np.insert(G, colIndex, newCol, axis=1)
    return Gnew


# This adds a new grid value to G
def addGridValueToG(xvec, newVal, h, driftfun, difffun, G, rowIndex):
    G = addRowToG(xvec, newVal, h, driftfun, difffun, G, rowIndex)
    G = addColumnToG(xvec, newVal, h, driftfun, difffun, G, rowIndex)
    return G


# This removes a new grid value from G
def removeGridValuesFromG(xValIndexToRemove, G):
    G = np.delete(G, xValIndexToRemove, 0)
    G = np.delete(G, xValIndexToRemove, 1)
    return G


# Adds the new value to the xvec grid in the correct location based on numerical order
def addValueToXvec(xvec, newVal):
    xnewLoc = np.searchsorted(xvec, newVal)
    xvec_new = np.insert(xvec, xnewLoc, newVal)
    return xnewLoc, xvec_new


# Check if we should remove values from G because they are "zero"
def checkReduceG(G, phat):
    integrandMaxes = Integrand.computeIntegrandArray(G, phat)
    integrandMaxes[(integrandMaxes < machEps) & (phat < machEps)] = -np.inf
    return integrandMaxes


# visualization parameters
finalGraph = False
animate = True
plotEvolution = False
plotEps = False
animateIntegrand = True
plotGSizeEvolution = True
plotLargestEigenvector = True

# tolerance parameters
epsilonTolerance = -20
minSizeGAndStillRemoveValsFromG = 60
minMaxOfPhatAndStillRemoveValsFromG = 0.01

# Run parameters
autoCorrectInitialGrid = True
RemoveFromG = True
AddToG= True


# simulation parameters
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

# pdf after one time step with Dirac delta(x-init) initial condition
a = init + driftfun(init)
b = np.abs(difffun(init)) * np.sqrt(h)

if not autoCorrectInitialGrid:
    xvec = np.arange(xMin, xMax, k)
    phat = dnorm(xvec, a, b)

if autoCorrectInitialGrid:
    # Figure out better initial max and min for the grid.
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

# Kernel matrix
G = integrandmat(xvec, xvec, h, driftfun, difffun)

# plt.figure()
# plt.plot(xvec, phat)
# plt.show()

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
            valsToRemove = checkReduceG(G, pdf)  # Remove if val is -inf
            if -np.inf in valsToRemove:
                for ind_w in reversed(range(len(valsToRemove))):  # reversed to avoid index problems
                    if (valsToRemove[ind_w] == -np.inf) & (
                            np.max(pdf > minMaxOfPhatAndStillRemoveValsFromG)) & (len(G) > minSizeGAndStillRemoveValsFromG):
                        G = removeGridValuesFromG(ind_w, G)
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
                G = addGridValueToG(xvec, leftEnd, h, driftfun, difffun, G, 0)
                xLoc, xvec = addValueToXvec(xvec, leftEnd)
                pdf = np.insert(pdf, xLoc, 0)
                G = addGridValueToG(xvec, rightEnd, h, driftfun, difffun, G, len(G))
                xLoc, xvec = addValueToXvec(xvec, rightEnd)
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
            pdf_trajectory.append(np.dot(G * k, pdf))
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

def scaleEigenvector(eigenvector, stepSizes):
    scale = np.real(np.matmul(eigenvector, stepSizes))
    scale = 1/scale
    return (scale)*eigenvector

if plotLargestEigenvector:
    vals, vects = np.linalg.eig(G_history[-1])
    largest_eigenvector = scaleEigenvector(vects[:, 0], k*np.ones(len(vects[:, 0])))
    plt.figure()
    plt.plot(xvec_trajectory[-1],pdf_trajectory[-1], label = 'PDF')
    plt.plot(xvec_trajectory[-1],largest_eigenvector, '.k', label = 'Eigenvector')  #blue
    plt.legend()
    plt.show()

