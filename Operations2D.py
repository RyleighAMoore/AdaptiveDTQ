import XGrid
import numpy as np
import Functions as fun
import Integrand

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
        
  

def generate2DG(inds_unrav, kstep, h, x1, x2):
    Gmat = np.zeros([len(inds_unrav[0]), len(inds_unrav[1])])
    for i in range(0, len(inds_unrav[0])): # I
        print(i)
        print(len(inds_unrav[0]))            
        for k in range(0, len(inds_unrav[0])): # K
            Gmat[i,k]=fun.G(x1[inds_unrav[0][i]], x2[inds_unrav[1][i]], x1[inds_unrav[0][k]], x2[inds_unrav[1][k]], h)
    return Gmat  



def updateGridExteriors2D(xvec,h, G, pdf, yvals, allXvals, yval,kstep, sliceNum):
    x1 = np.hstack(allXvals)
    x2 = np.hstack(yvals)
    x1star = np.vstack(allXvals).T # listing y values for column
    x1star = np.hstack(x1star)
    x2star = np.vstack(yvals).T # listing y values for column
    x2star = np.hstack(x2star)
    leftEnd = xvec[0] - (xvec[1] - xvec[0])
    rightEnd = xvec[-1] + (xvec[-1] - xvec[-2])
    allXvals[0].append(rightEnd)
    newRow_left = [] 
    newCol_left = []
    
    for k in range(0, len(x1)): # K
        newRow_left.append(kstep**2*fun.G(leftEnd, yval,x1[k], x2[k], h))
    
    for i in range(len(xvec)):
        newCol_left.append(kstep**2*fun.G(leftEnd, yval,x1[i], x2star[i], h))
    
    Gnew = np.insert(G, 0, newCol_left, axis=1)
    Gnew = np.insert(Gnew, -1, newRow_left, 0)
    xLoc, xvec = addValueToXvec(xvec, leftEnd)
    pdf = np.insert(pdf, xLoc, 0)
    G = GMatrix.addGridValueToG(xvec, rightEnd, h, G, len(G))
    xLoc, xvec = addValueToXvec(xvec, rightEnd)
    pdf = np.insert(pdf, xLoc, 0)
    epsilon = Integrand.computeEpsilon(G, G * pdf)
    return G, pdf,xvec,epsilon

    
def stepForwardInTime2D(xvec, pdf, G, init, h, kstep, yvals, allXvals, yval, sliceNum, phatList, runSum):
    epsilon = Integrand.computeEpsilon(G, pdf)
    print(epsilon)
    tol=-15
    if epsilon > tol:
        while epsilon >= tol:
            #############################################  adding to grid exterior
            print("adding grid")
            #G, pdf, xvec, epsilon = updateGridExteriors2D(xvec, h, G, pdf, yvals, allXvals, yval,k, sliceNum)
            leftEnd = xvec[0] - (xvec[1] - xvec[0])
            rightEnd = xvec[-1] + (xvec[-1] - xvec[-2])
            allXvals[sliceNum] = np.insert(allXvals[sliceNum],len(allXvals[sliceNum]), rightEnd)
            allXvals[sliceNum] = np.insert(allXvals[sliceNum],0, leftEnd)
            yvals[sliceNum] = np.insert(yvals[sliceNum],len(yvals[sliceNum]), yvals[sliceNum][0])
            yvals[sliceNum] = np.insert(yvals[sliceNum],0, yvals[sliceNum][0])
            phatList[sliceNum] = np.insert(phatList[sliceNum],len(phatList[sliceNum]), 0)
            phatList[sliceNum] = np.insert(phatList[sliceNum],0, 0)
            
            Ys = np.hstack(yvals)
            Xs = np.hstack(allXvals)
            pdf = np.hstack(phatList)
            Gmat = np.zeros([len(Ys), len(Ys)])
            for i in range(0, len(Ys)): # I
                for k in range(0, len(Ys)): # K
                    Gmat[i,k]=kstep**2*fun.G(Xs[i], Ys[i], Xs[k], Ys[k], h)
            
            startG = 0
            c=0
            while c < sliceNum:
                startG = startG + len(allXvals[c])
                c=c+1
            
            G = Gmat[startG:startG+len(allXvals[sliceNum]),:]
            epsilon = Integrand.computeEpsilon(G, pdf)            
            print(epsilon)
            ################################################

    if epsilon <= tol:  # things are going well
        pdf = np.dot(G, pdf)  # Equispaced Trapezoidal Rule
        
        return allXvals[sliceNum], yvals[sliceNum], pdf

#x1 = XGrid.getCenteredZeroXvec(0.15, 3)
#x2 = XGrid.getCenteredZeroXvec(0.15, 3)
#
#w = find_nearest(x1, 0)
#phat = np.zeros([len(x1), len(x2)])
#phat0 = fun.dnorm(x1, 0, 0.1)  # pdf after one time step with Dirac \delta(x-init)
#phat[:, w] = phat0
#
#inds = np.asarray(list(range(0, len(x1)*len(x2))))
#phat_rav = np.ravel(phat.T)
#
#inds_unrav = np.unravel_index(inds, (len(phat), len(phat)))

