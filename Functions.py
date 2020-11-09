import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import erf

'''Simple'''
# def f1(x, y):
#     return 0

# def f2(x, y):
#     return 0

# def g1(x=0,y=0):
#     return 1

# def g2(x=0,y=0):
#     return 1

'''Erf'''
def drift(mesh):
    x = mesh[:,0]
    y = mesh[:,1]
    r = np.sqrt(x ** 2 + y ** 2)
    # return 1
    # return np.asarray([10*x*(1- r ** 2), 10*y*(1- r ** 2)]).T

    return np.asarray([5*erf(10*x), np.zeros((np.size(mesh,0)))]).T
# def f1(x, y):
#     return 5

# def f2(x, y):
#     return 0
def diff(mesh):
    if mesh.ndim ==1:
        mesh = np.expand_dims(mesh, axis=0)
    x = mesh[:,0]
    y = mesh[:,1]
    return np.diag([1,1])
    

# def g1(x=0,y=0):
#     return 1

# def g2(x=0,y=0):
#     return 1
    
    
    
# Drift fuction
def driftfun(x):
#     if isinstance(x, int) | isinstance(x, float):
#         return 1
#     else:
#         return np.ones(np.shape(x)) * 1
    return x * (4 - x ** 2)

# Diffusion ction
def difffun(x):
    return np.repeat(1, np.size(x))


# Density, distribution ction, quantile ction and random generation for the
# normal distribution with mean equal to mu and standard deviation equal to sigma.
def dnorm(x, mu, sigma):
    return np.divide(1, (sigma * np.sqrt(2 * np.pi))) * np.exp(np.divide(-(x - mu) ** 2, 2 * sigma ** 2))


def dnorm_partialx(x, mu, sigma):
    return np.divide(-x+mu,sigma**2)*np.divide(1, (sigma * np.sqrt(2 * np.pi))) * np.exp(np.divide(-(x - mu) ** 2, 2 * sigma ** 2))


def GVals(indexOfMesh, mesh, h):
    val = G(indexOfMesh,mesh,h)
    return val

# def HVals(x1,x2,mesh,h):
#     y1 = mesh[:,0]
#     y2 = mesh[:,1]
#     scale = h*g1(x1,x2)*g2(x1,x2)/(h*g1(y1,y2)*g2(y1,y2))
#     val = scale*np.exp(-(h**2*f1(y1,y2)**2+2*h*f1(y1,y2)*(x1-y1))/(2*h*g1(x1,x2)**2) + -(h**2*f2(y1,y2)**2+2*h*f2(y1,y2)*(x2-y2))/(2*h*g2(x1,x2)**2))*np.exp((x1-y1+h*f1(y1,y2))**2/(2*h*g1(x1,x2)**2) - (x1-y1+h *f1(y1,y2))**2/(2*h*g1(y1,y2)**2) + (x2-y2+h*f2(y1,y2))**2/(2*h*g2(x1,x2)**2) - (x2-y2+h* f2(y1,y2))**2/(2*h*g2(y1,y2)**2))
#     val = scale*np.exp(-(h**2*f1(y1,y2)**2-2*h*f1(y1,y2)*(x1-y1))/(2*h*g1(x1,x2)**2) + -(h**2*f2(y1,y2)**2-2*h*f2(y1,y2)*(x2-y2))/(2*h*g2(x1,x2)**2))*np.exp((x1-y1-h*f1(y1,y2))**2/(2*h*g1(x1,x2)**2) - (x1-y1-h *f1(y1,y2))**2/(2*h*g1(y1,y2)**2) + (x2-y2-h*f2(y1,y2))**2/(2*h*g2(x1,x2)**2) - (x2-y2-h* f2(y1,y2))**2/(2*h*g2(y1,y2)**2))
#     return val

def Gaussian(scaling, mesh):
    rv = multivariate_normal(scaling.mu.T[0], scaling.cov)        
    soln_vals = np.asarray([rv.pdf(mesh)]).T
    return np.squeeze(soln_vals)


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
    cov = diff(mesh) ** 2 * h
    soln_vals = []
    const = 1/(np.sqrt((2*np.pi)**D*abs(np.linalg.det(cov))))

    for j in range(len(mesh)):
        mu = mean[j,:]
        Gs = np.exp(-1/2*((x-mu).T@np.linalg.inv(cov)@(x.T-mu.T)))
        soln_vals.append(Gs*const)

    return np.asarray(soln_vals)

def AddPointToG(mesh, newPointindex, h, GMat):
    newRow = G(newPointindex, mesh,h)
    GMat[newPointindex,:len(newRow)] = newRow

    D = mesh.shape[1]
    mean = mesh[-1,:]+drift(np.expand_dims(mesh[-1,:],axis=0))*h
    mean = mean[0]
    cov = diff(mesh) ** 2 * h
    newCol = []
    const = 1/(np.sqrt((2*np.pi)**D*abs(np.linalg.det(cov))))
    
    for j in range(len(mesh)):
        mu = mean
        x = mesh[j,:]
        Gs = np.exp(-1/2*((x-mu).T@np.linalg.inv(cov)@(x.T-mu.T)))
        newCol.append(Gs*const)
    
    GMat[:len(newCol),newPointindex] = newCol
    return GMat
    
    
    
    
    
    
    
    