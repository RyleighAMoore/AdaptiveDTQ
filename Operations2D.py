import XGrid
import numpy as np
import Functions as fun

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def computeEpsilon(Gmat, phat_rav): #TODO Look at this method... Not sure that this is quite right
    tol = 0
    M,N = Gmat.shape
    s = int(np.sqrt(N))
    M = int(M)
    N=int(N)
    firstRow = np.max((Gmat[0:s, :].T * phat_rav[0:s]).T)
    lastRow = np.max((Gmat[-s:,:].T * phat_rav[-s:]).T)
    firstCol = np.max((Gmat[0::s].T * phat_rav[0::s]).T)
    lastCol = np.max((Gmat[s-1::s].T * phat_rav[s-1::s]).T)
    vals = np.asarray(np.log([firstRow, lastRow, firstCol, lastCol]))
    if (firstRow <= tol) & (lastRow <= tol) & (firstCol <= tol) & (lastCol <= tol):
        val = -np.inf
    else:
        val = np.log(max(firstRow, lastRow, firstCol, lastCol))
    return val, vals


def addValToG(x1Bool, x1, x2, loc, val, h):
    x1loc = np.searchsorted(x1, val)
    x2loc = np.searchsorted(x1, val)
    for k in range(0, len(inds_unrav[0])): # K
        Gmat[i,k]=kstep**2*G(x1[inds_unrav[0][i]], x2[inds_unrav[1][i]], x1[inds_unrav[0][k]], x2[inds_unrav[1][k]], h)
        
        
        

x1 = XGrid.getCenteredZeroXvec(0.15, 3)
x2 = XGrid.getCenteredZeroXvec(0.15, 3)

w = find_nearest(x1, 0)
phat = np.zeros([len(x1), len(x2)])
phat0 = fun.dnorm(x1, 0, 0.1)  # pdf after one time step with Dirac \delta(x-init)
phat[:, w] = phat0

inds = np.asarray(list(range(0, len(x1)*len(x2))))
phat_rav = np.ravel(phat.T)

inds_unrav = np.unravel_index(inds, (len(phat), len(phat)))

