import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import erf

def drift(mesh):
    if mesh.ndim ==1:
        mesh = np.expand_dims(mesh, axis=0)
    x = mesh[:,0]
    y = mesh[:,1]
    r = np.sqrt(x ** 2 + y ** 2)
    # return np.asarray([x**2/2-y*x, x*y+y**2/2]).T
    # return np.asarray([x-y,x+y]).T
    return np.asarray([np.ones((np.size(mesh,0))), np.zeros((np.size(mesh,0)))]).T
    # return np.asarray([5*x*(3- r ** 2), 5*y*(3- r ** 2)]).T
    return np.asarray([3*erf(10*x), 3*erf(10*y)]).T

def diff(mesh):
    if mesh.ndim == 1:
        mesh = np.expand_dims(mesh, axis=0)
    # return np.diag([1,1]) + np.ones((2,2))*0.5
    # return np.diag([mesh[:,0][0],mesh[:,1][0]])
    return np.diag([1,1])

    return np.diag([0.75,0.75])

# Density, distribution ction, quantile ction and random generation for the
# normal distribution with mean equal to mu and standard deviation equal to sigma.
def dnorm(x, mu, sigma):
    return np.divide(1, (sigma * np.sqrt(2 * np.pi))) * np.exp(np.divide(-(x - mu) ** 2, 2 * sigma ** 2))


def dnorm_partialx(x, mu, sigma):
    return np.divide(-x+mu,sigma**2)*np.divide(1, (sigma * np.sqrt(2 * np.pi))) * np.exp(np.divide(-(x - mu) ** 2, 2 * sigma ** 2))


def GVals(indexOfMesh, mesh, h):
    val = G(indexOfMesh,mesh,h)
    return val


def Gaussian(scaling, mesh):
    rv = multivariate_normal(scaling.mu.T[0], scaling.cov)        
    soln_vals = np.asarray([rv.pdf(mesh)]).T
    return np.squeeze(soln_vals)

def weightExp(scaling,mesh):
    mu = scaling.mu
    D = mesh.shape[1]
    cov = scaling.cov
    soln_vals = np.empty(len(mesh))
    # const = 1/(np.sqrt((np.pi)**D*abs(np.linalg.det(cov))))
    invCov = np.linalg.inv(cov)
    for j in range(len(mesh)):
        x = np.expand_dims(mesh[j,:],1)
        Gs = np.exp(-(x-mu).T@invCov@(x-mu))
        soln_vals[j] = Gs
    return soln_vals


def covPart(Px, Py, mesh, cov):
    vals = []
    for i in range(len(mesh)):
        val = np.exp(-2*cov*(mesh[i,0]-Px)*(mesh[i,1]-Py))
        vals.append(np.copy(val))
    return np.asarray(vals)


def G(indexOfMesh,mesh, h):
    x = mesh[indexOfMesh,:]
    D = mesh.shape[1]
    mean = mesh+drift(mesh)*h
    # cov = diff(mesh) ** 2 * h
    cov = diff(mesh)@diff(mesh).T * h
    soln_vals = np.empty(len(mesh))
    const = 1/(np.sqrt((2*np.pi)**D*abs(np.linalg.det(cov))))
    invCov = np.linalg.inv(cov)
    for j in range(len(mesh)):
        mu = mean[j,:]
        Gs = np.exp(-1/2*((x-mu).T@invCov@(x.T-mu.T)))
        soln_vals[j] = Gs
    return soln_vals*const




def AddPointToG(mesh, newPointindex, h, GMat):
    newRow = G(newPointindex, mesh,h)
    GMat[newPointindex,:len(newRow)] = newRow
    D = mesh.shape[1]
    mu = mesh[-1,:]+drift(np.expand_dims(mesh[-1,:],axis=0))*h
    mu = mu[0]
    # cov = diff(mesh) ** 2 * h # put inside loop if cov changes spatially
    cov = diff(mesh)@diff(mesh).T*h
    newCol = np.empty(len(mesh))
    const = 1/(np.sqrt((2*np.pi)**D*abs(np.linalg.det(cov))))
    covInv = np.linalg.inv(cov)
    for j in range(len(mesh)):
        x = mesh[j,:]
        Gs = np.exp(-1/2*((x-mu).T@covInv@(x.T-mu.T)))
        newCol[j] = (Gs)

    GMat[:len(newCol),newPointindex] = newCol*const
    return GMat



# Drift fuction
def driftfun(x):
    if isinstance(x, int) | isinstance(x, float):
        return 1
    else:
        return np.ones(np.shape(x)) * 1
    # return x * (4 - x ** 2)

# Diffusion ction
def difffun(x):
    return np.repeat(1, np.size(x))
