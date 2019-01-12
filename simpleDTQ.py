# code to compute the PDF of the solution of the SDE:
#
# dX_t = X_t*(4-X_t^2) dt + dW_t
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Drift function
def driftfun(x):
    return x * (4 - x ** 2)


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
animate = True

# simulation parameters
T = 1  # final time, code computes PDF of X_T
s = 0.75  # the exponent in the relation k = h^s
h = 0.01  # temporal step size
init = 0  # initial condition X_0
numsteps = int(np.ceil(T / h))

assert numsteps > 0, 'The variable numsteps must be greater than 0'

# define spatial grid
k = h ** s
xMin = 5
xMax = 5
xvec = np.arange(-xMin, xMax, k)

# Kernel matrix
A = np.multiply(k, integrandmat(xvec, xvec, h, driftfun, difffun))

# pdf after one time step with Dirac delta(x-init) initial condition
phat = dnorm(xvec, init + driftfun(init), np.abs(difffun(init)) * np.sqrt(h))

if animate:
    pdf_trajectory = np.zeros([phat.size, numsteps+1])
    pdf_trajectory[:,0] = phat
    for i in range(numsteps-1):  # since one time step is computed above
        pdf_trajectory[:,i+1] = np.dot(A, pdf_trajectory[:,i])

    def update_animation(step, pdf_data, l):
        l.set_xdata(xvec)
        l.set_ydata(pdf_data[:,step+1])
        return l,

    f1 = plt.figure()
    l, = plt.plot([],[],'r')
    plt.xlim(np.min(xvec), np.max(xvec))
    plt.ylim(0, np.max(phat))
    anim = animation.FuncAnimation(f1, update_animation, numsteps, fargs=(pdf_trajectory,l), interval=50, blit=True)

    plt.show()

else:
    # main iteration loop
    for i in range(numsteps-1):  # since one time step is computed above
        phat = np.matmul(A, phat)

    plt.plot(xvec, phat, '.')
    plt.show()