# code to compute the PDF of the solution of the SDE:
#
# dX_t = X_t*(4-X_t^2) dt + dW_t
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import Integrand
from matplotlib.animation import FuncAnimation

# Drift function
def driftfun(x):
    return x*(40- x ** 2)


# Diffusion function
def difffun(x):
    return np.repeat(1, np.size(x))


def dnorm(x, mu, sigma):
    return np.divide(1, (sigma * np.sqrt(2 * np.pi))) * np.exp(np.divide(-(x - mu) ** 2, 2 * sigma ** 2))


#  Function that returns the kernel matrix G(x,y)
def integrandmat(xvec, yvec, h, driftfun, difffun):
    Y = np.zeros((len(yvec), len(yvec)))
    for i in range(len(yvec)):
        Y[i, :] = xvec  # Y has the same grid value along each column (col1 has x1, col2 has x2, etc)
    mu = Y + driftfun(Y) * h
    sigma = abs(difffun(Y)) * np.sqrt(h)
    sigma = np.reshape(sigma, [np.size(xvec), np.size(xvec)])  # make a matrix for the dnorm function
    Y = np.transpose(Y)  # Transpose Y for use in the dnorm function
    test = dnorm(Y, mu, sigma)
    return test

# This adds a N dimensional row to a M by N dimensional G
def addRowToG(xvec, newVal, h, driftfun, difffun, G, rowIndex):
    mu = xvec + driftfun(xvec) * h
    sigma = abs(difffun(xvec)) * np.sqrt(h)
    xrep = np.ones(len(mu))*newVal
    newRow = dnorm(xrep, mu, sigma)
    Gnew = np.insert(G, rowIndex, newRow, 0)
    return Gnew

# This adds a M dimensional column to a M by N dimensional G
def addColumnToG(xvec, newVal, h, driftfun, difffun, G, colIndex):
    mu = np.ones(len(G))*(newVal + driftfun(newVal) * h)
    w = np.ones(len(G))*newVal
    sigma = abs(difffun(w)) * np.sqrt(h)
    xnewLoc = np.searchsorted(xvec,newVal)
    xnew = np.insert(xvec, xnewLoc, newVal)
    newCol = dnorm(xnew, mu, sigma)
    Gnew = np.insert(G, colIndex, newCol, axis=1)
    return Gnew


def addGridValueToG(xvec, newVal, h, driftfun, difffun, G, rowIndex):
    G = addRowToG(xvec, newVal, h, driftfun, difffun, G, rowIndex)
    G = addColumnToG(xvec, newVal, h, driftfun, difffun, G, rowIndex)
    return G

def removeGridValuesFromG(xValIdexToRemove, G):
    G = np.delete(G, xValIdexToRemove, 0)
    G = np.delete(G, xValIdexToRemove, 1)
    return G


def addValueToXvec(xvec, newVal):
    xnewLoc = np.searchsorted(xvec, newVal)
    xvec_new = np.insert(xvec, xnewLoc, newVal)
    return xnewLoc, xvec_new

def checkReduceG(G, phat):
    tol = 0.0000001
    tol2 = 0.00001
    integradMaxes = Integrand.computeIntegrandArray(G,phat)
    integradMaxes[(integradMaxes < tol) & (phat < tol2)] = -np.inf
    return integradMaxes


# visualization parameters
finalGraph = False
animate = True
plotEvolution = False
plotEps = False
animateIntegrand = False

# simulation parameters
T = 0.5  # final time, code computes PDF of X_T
s = 0.75  # the exponent in the relation k = h^s
h = 0.01  # temporal step size
init = 0  # initial condition X_0
numsteps = int(np.ceil(T / h))

assert numsteps > 0, 'The variable numsteps must be greater than 0'

# define spatial grid
k = h ** s
#k = 0.1
xMin = -1
xMax = 1
xvec = np.arange(xMin, xMax, k)

# Kernel matrix
G = integrandmat(xvec, xvec, h, driftfun, difffun)
Gk = np.multiply(k, G)

# pdf after one time step with Dirac delta(x-init) initial condition
phat = dnorm(xvec, init + driftfun(init), np.abs(difffun(init)) * np.sqrt(h))

pdf_trajectory = []
xvec_trajectory = []
epsilonArray = []
G_history = []
epsilonArray.append(Integrand.computeEpsilon(G, phat))
if animate:
    pdf_trajectory.append(phat)  # solution after one time step from above
    xvec_trajectory.append(xvec)
    if animateIntegrand: G_history.append(G)
    countSteps = 0
    while countSteps < numsteps-1:  # since one time step is computed above
        epsilon = Integrand.computeEpsilon(G, pdf_trajectory[-1])
        tol = -10
        if epsilon <= tol:
            valsToRemove = checkReduceG(G, pdf_trajectory[-1])  # Remove if val is -inf
            if (len(G) > 60) & (countSteps > 1) & (-np.inf in valsToRemove):
                xvec_trajectory.append(xvec_trajectory[-1])
                pdf_trajectory.append(pdf_trajectory[-1])
                for ind_w in reversed(range(len(valsToRemove))):
                    if valsToRemove[ind_w] == -np.inf:
                        G = removeGridValuesFromG(ind_w, G)
                        xvec_trajectory[-1] = np.delete(xvec_trajectory[-1], ind_w)
                        pdf_trajectory[-1] = np.delete(pdf_trajectory[-1], ind_w)
                countSteps = countSteps + 1
                if animateIntegrand: G_history.append(G)
            else:
                pdf_trajectory.append(np.dot(G*k, pdf_trajectory[-1]))
                xvec_trajectory.append(xvec_trajectory[-1])
                if animateIntegrand: G_history.append(G)
                epsilonArray.append(Integrand.computeEpsilon(G, pdf_trajectory[-1]))
                countSteps = countSteps+1
            test = 0
        else:
            IC = False
            if len(xvec_trajectory) < 2: # pdf trajectory size is 1
                flag = True
            while epsilon >= tol:
                ############################################## adding to grid
                leftEnd = xvec_trajectory[-1][0] - k
                rightEnd = xvec_trajectory[-1][-1] + k
                G = addGridValueToG(xvec_trajectory[-1], leftEnd, h, driftfun, difffun, G, 0)
                xLoc, xvec_trajectory[-1] = addValueToXvec(xvec_trajectory[-1], leftEnd)
                pdf_trajectory[-1] = np.insert(pdf_trajectory[-1], xLoc, 0)
                G = addGridValueToG(xvec_trajectory[-1], rightEnd, h, driftfun, difffun, G, len(G))
                xLoc, xvec_trajectory[-1] = addValueToXvec(xvec_trajectory[-1], rightEnd)
                pdf_trajectory[-1] = np.insert(pdf_trajectory[-1], xLoc, 0)
                epsilon = Integrand.computeEpsilon(G, G*pdf_trajectory[-1])
                epsilonArray.append(epsilon)
                print(epsilon)
                ##############################################

            # recompute ICs
            if IC: pdf_trajectory[-1] = dnorm(xvec, init + driftfun(init), np.abs(difffun(init)) * np.sqrt(h))
            else: pdf_trajectory[-1] = (np.dot(G * k, pdf_trajectory[-1]))
            if animateIntegrand: G_history[-1] = G

    multiplier = 1.5
    minxgrid = np.floor(np.min(xvec_trajectory[0])) * multiplier
    maxxgrid = np.ceil(np.max(xvec_trajectory[0])) * multiplier

    def update_animation(step, pdf_traj, l):
        global minxgrid
        global maxxgrid
        if step == 0:
            minxgrid = np.floor(np.min(xvec_trajectory[0]) * multiplier)
            maxxgrid = np.ceil(np.max(xvec_trajectory[0]) * multiplier)
            plt.xlim(np.min(xvec_trajectory[0]) * multiplier, np.max(xvec_trajectory[0]) * multiplier)

        im.set_xdata(xvec_trajectory[step])
        im.set_ydata(pdf_traj[step])
        m = np.floor(np.min(xvec_trajectory[step]))
        M = np.ceil(np.max(xvec_trajectory[step]))
        if (m < minxgrid) & (M > maxxgrid):
            l.set_xlim(m * multiplier, M*multiplier)
            minxgrid = m
            maxxgrid = M

        elif m < minxgrid:
            l.set_xlim(m*multiplier, maxxgrid)
            minxgrid = m

        elif M > maxxgrid:
            l.set_xlim(minxgrid, M*multiplier)
            maxxgrid = M
        return im


    f1 = plt.figure()
    l = f1.add_subplot(1,1,1)
    im, = l.plot([],[],'r')
    plt.xlim(np.min(xvec_trajectory[0])*5, np.max(xvec_trajectory[0])*5)
    plt.ylim(0, np.max(phat))
    anim = animation.FuncAnimation(f1, update_animation, len(xvec_trajectory), fargs=(pdf_trajectory,l), interval=50, blit=False)
    plt.show()

if animateIntegrand:
    assert animate == True, 'Animate must be True'

    def update_animation_integrand(step, val, l):
        integrand = Integrand.calculateIntegrand(G_history[step], pdf_trajectory[step])
        Y = np.zeros([np.size(xvec_trajectory[step]), np.size(integrand,1)])
        for i in range(np.size(integrand,1)):
            Y[i, :] = xvec_trajectory[step]
        l.set_xdata(Y)
        l.set_ydata(integrand)
        return l,

    f1 = plt.figure()
    l, = plt.plot([],[],'r')
    plt.xlim(np.min(xvec_trajectory[-1]), np.max(xvec_trajectory[-1]), '.')
    plt.ylim(0, 14)
    anim = animation.FuncAnimation(f1, update_animation_integrand, numsteps, fargs=(3,l), interval=50, blit=True)
    plt.show()

if plotEps:
    assert animate == True, 'The variable animate must be True'
    plt.figure()
    plt.plot(epsilonArray)
    plt.xlabel('Time Step')
    plt.ylabel(r'$\varepsilon$ value')
    plt.title(r'$\varepsilon$ at each time step for $f(x)=x(4-x^2), g(x)=1, k \approx 0.032$, starting interval [-1,1], tol = -100')
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
    plt.plot(xvec_trajectory[int(np.ceil(numPDF*(1/3)))], pdf_trajectory[int(np.ceil(numPDF*(1/3)))])
    plt.subplot(2, 2, 3)
    plt.title("t=2T/3")
    plt.plot(xvec_trajectory[int(np.ceil(numPDF*(2/3)))], pdf_trajectory[int(np.ceil(numPDF * (2 / 3)))])
    plt.subplot(2, 2, 4)
    plt.title("t=T")
    plt.plot(xvec_trajectory[int(np.ceil(numPDF-1))], pdf_trajectory[int(np.ceil(numPDF-1))])
    plt.show()


if finalGraph:
    assert animate == True, 'The variable animate must be True'
    plt.plot(xvec_trajectory[-1], pdf_trajectory[-1], '.')
    plt.show()


plt.figure()

'''
for j,i in enumerate(G_history):
    w = i.shape[0]
    plt.plot(j, w,'.')
plt.title(r'Size of G at each time step for $f(x)=x(4-x^2), g(x)=1, k \approx 0.032$, starting interval [-1,1], tol = -100')
plt.show()
'''
