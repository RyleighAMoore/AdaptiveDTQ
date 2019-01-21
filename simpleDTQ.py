# code to compute the PDF of the solution of the SDE:
#
# dX_t = X_t*(4-X_t^2) dt + dW_t
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import Integrand

# Drift function
def driftfun(x):
    return (x * (4 - x ** 2))


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
    Y = np.transpose(Y)  # Transpose Y for use in the dnorm function
    sigma = abs(difffun(Y)) * np.sqrt(h)
    sigma = np.reshape(sigma, [np.size(xvec), np.size(xvec)])  # make a matrix for the dnorm function
    test = dnorm(Y, mu, sigma)
    return test


# visualization parameters
finalGraph = False
animate = True
plotEvolution = False
saveSolution = False
gridFileName = 'CoarseX'
solutionFileName = 'CoarseSolution'
plotEps = True
animateIntegrand = True

# simulation parameters
T = 1  # final time, code computes PDF of X_T
s = 0.75  # the exponent in the relation k = h^s
h = 0.01  # temporal step size
init = 0  # initial condition X_0
numsteps = int(np.ceil(T / h))

assert numsteps > 0, 'The variable numsteps must be greater than 0'

# define spatial grid
k = h ** s
# k = 0.1
xMin = - 4
xMax = 4
xvec = np.arange(xMin, xMax, k)

# Kernel matrix
G = integrandmat(xvec, xvec, h, driftfun, difffun)
A = np.multiply(k, G)

# pdf after one time step with Dirac delta(x-init) initial condition
phat = dnorm(xvec, init + driftfun(init), np.abs(difffun(init)) * np.sqrt(h))

pdf_trajectory = np.zeros([phat.size, numsteps])
epsilon = np.zeros(numsteps)
epsilon[0] = Integrand.computeEpsilon(G, phat)
if animate:
    pdf_trajectory[:, 0] = phat  # solution after one time step from above
    for i in range(numsteps-1):  # since one time step is computed above
        pdf_trajectory[:,i+1] = np.dot(A, pdf_trajectory[:,i])
        epsilon[i+1] = Integrand.computeEpsilon(G, pdf_trajectory[:,i+1])

    def update_animation(step, pdf_data, l):
        l.set_xdata(xvec)
        l.set_ydata(pdf_data[:,step])
        return l,

    f1 = plt.figure()
    l, = plt.plot([],[],'r')
    plt.xlim(np.min(xvec), np.max(xvec))
    plt.ylim(0, np.max(phat))
    anim = animation.FuncAnimation(f1, update_animation, numsteps, fargs=(pdf_trajectory,l), interval=50, blit=True)
    plt.show()

phat_history = np.zeros([phat.size, numsteps])
if animateIntegrand:
    phat_history[:,0] = phat
    for i in range(numsteps-1):  # since one time step is computed above
        phat_history[:, i+1] = np.dot(A, phat_history[:, i])

    def update_animation(step, Y, l):
        integrand = Integrand.calculateIntegrand(G, phat_history[:,step])
        l.set_xdata(Y)
        l.set_ydata(integrand)
        return l,

    f1 = plt.figure()
    l, = plt.plot([],[],'r')
    plt.xlim(np.min(xvec), np.max(xvec))
    plt.ylim(0, np.max(G*phat))
    Y = np.zeros((len(xvec), len(xvec)))  # Matrix with xvec along each column for use in animation
    for i in range(len(xvec)):
        Y[i, :] = xvec
    anim = animation.FuncAnimation(f1, update_animation, numsteps, fargs=(Y,l), interval=50, blit=True)
    plt.show()

if plotEps:
    assert animate == True, 'The variable animate must be True'
    plt.figure()
    plt.plot(epsilon)
    plt.xlabel('Time Step')
    plt.ylabel(r'$\varepsilon$ value')
    plt.title(r'$\varepsilon$ at each time step for $f(x)=x(4-x^2), g(x)=1, k \approx 0.032$, interval [-4,4]')
    plt.show()

if plotEvolution:
    plt.figure()
    plt.suptitle(r'Evolution for $f(x)=x(4-x^2), g(x)=1, k \approx 0.032$')
    numPDF = np.size(pdf_trajectory,1)
    plt.subplot(2, 2, 1)
    plt.title("t=0")
    plt.plot(xvec, pdf_trajectory[:,0])
    plt.subplot(2, 2, 2)
    plt.title("t=T/3")
    plt.plot(xvec, pdf_trajectory[:, int(np.ceil(numPDF*(1/3)))])
    plt.subplot(2, 2, 3)
    plt.title("t=2T/3")
    plt.plot(xvec, pdf_trajectory[:, int(np.ceil(numPDF * (2 / 3)))])
    plt.subplot(2, 2, 4)
    plt.title("t=T")
    plt.plot(xvec, pdf_trajectory[:, int(np.ceil(numPDF-1))])
    plt.show()

if saveSolution:
    SolutionToSave = open(solutionFileName, 'wb')
    pickle.dump(pdf_trajectory, SolutionToSave)
    SolutionToSave.close()
    GridToSave = open(gridFileName, 'wb')
    pickle.dump(xvec, GridToSave)
    GridToSave.close()

if finalGraph:
    # main iteration loop
    for i in range(numsteps-1):  # since one time step is computed above
        phat = np.matmul(A, phat)
    plt.plot(xvec, phat, '.')
    plt.show()


