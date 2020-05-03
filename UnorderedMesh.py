import numpy as np
import XGrid
import random
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import Functions as fun
import Operations2D
from mpl_toolkits.mplot3d import Axes3D
# xCoord, yCoord, is the point we are looking for the closests 
# two points to. 
# AllPoints is a Nx2  matrix of all the degrees of freedom.
#  x1   y1
#  x2   y2
#  x3   y3
#  ...  ...
def findNearestKPoints(xCoord, yCoord, AllPoints, numNeighbors, getIndices = False):
    normList = []
    for point in range(len(AllPoints)):
        xVal = AllPoints[point,0]
        yVal = AllPoints[point,1]
        normList.append(np.sqrt((xCoord-xVal)**2+(yCoord-yVal)**2))
    idx = np.argsort(normList)
    neighbors = []
    distances = []
    indices = []
    for k in range(1,numNeighbors+1):
        neighbors.append(np.asarray([AllPoints[idx[k],0], AllPoints[idx[k],1]]))
        distances.append(normList[idx[k]])
        assert normList[idx[k]] > 0, "point wrong"
        indices.append(idx[k])
    neighbors = np.asarray(neighbors)
    # if len(neighbors) > 0:
    #     plt.figure()
    #     plt.plot(AllPoints[:,0], AllPoints[:,1], '.')
    #     plt.plot(xCoord,yCoord, '*r')
    #     plt.plot(neighbors[:,0], neighbors[:,1],'.')
    #     plt.show()
    # print(distances)
    if getIndices:
        return neighbors, distances, indices
    return neighbors, distances

# neighbors, distances = findNearestKPoints(-1.7, 1, Meshes[0], 1)

def findNearestPoint(xCoord, yCoord, AllPoints, includeIndex=False, samePointRet0 = False):
    normList = []
    for point in range(len(AllPoints)):
        xVal = AllPoints[point,0]
        yVal = AllPoints[point,1]
        normList.append(np.sqrt((xCoord-xVal)**2+(yCoord-yVal)**2))
    idx = np.argsort(normList)
    #print(idx)
    if normList[idx[0]] == 0: # point is part of the set
        if includeIndex == True:
            try:
                return np.asarray([[AllPoints[idx[1],0], AllPoints[idx[1],1]]]), idx[1]
            except: 
                t=0
        else:
            if samePointRet0:
               return np.asarray([[AllPoints[idx[0],0], AllPoints[idx[0],1]]])
            else:
                return np.asarray([[AllPoints[idx[0],0], AllPoints[idx[0],1]]]), idx[0]
    else:
        if includeIndex:
            return np.asarray([[AllPoints[idx[0],0], AllPoints[idx[0],1]]]), idx[0]
        else:
            return np.asarray([[AllPoints[idx[0],0], AllPoints[idx[0],1]]]), idx[0]
    

# point: Point to center grid around
# spacing: step size of grid
# span: Units around the point to make the grid.
def makeOrderedGridAroundPoint(point, spacing, span, xmin, xmax, ymin, ymax):
    span = 2
    percent = 0.01
    x = XGrid.getCenteredXvecAroundPoint(spacing, span, span, point[0])
    y = XGrid.getCenteredXvecAroundPoint(spacing, span, span, point[1])
    x = np.asarray(x)
    y = np.asarray(y)
    x = x[(x>=xmin+percent*(xmax-xmin)) & (x<=xmax-percent*(xmax-xmin))]
    y = y[(y>=ymin+percent*(xmax-xmin)) & (y<=ymax-percent*(xmax-xmin))]
#    if (min(x)<xmin) | (max(x)>xmax)| (min(y)<ymin) | (max(y)>ymax):
#        print('Problem in making grid')
    X, Y = np.meshgrid(x, y)
    xVec = np.reshape(X,[1,np.size(X)],order='C').T
    yVec = np.reshape(Y,[1,np.size(Y)],order='C').T
    points = np.hstack((xVec,yVec))
    return points

# grid = makeOrderedGridAroundPoint(point, spacing, span, xmin, xmax, ymin, ymax)


# generate random points from [xMin,xMax] x [yMin, yMax]
# returns 
#  x1   y1
#  x2   y2
#  x3   y3
#  ...  ...
def generateRandomPoints(xMin,xMax,yMin,yMax, numPoints):
    x1 =[]
    x2=[]
    for i in range(numPoints):
        x1.append(random.uniform(xMin, xMax))
        x2.append(random.uniform(yMin, yMax))
        
    return np.asarray([x1,x2]).T

    
def generateICPDF(x,y,sigma_x, sigma_y):
    z = (1/(2*np.pi*sigma_x*sigma_y) * np.exp(-(x**2/(2*sigma_x**2)
         + y**2/(2*sigma_y**2))))    
    return z

def generateICPDFShifted(x,y,sigma_x, sigma_y, muX, muY):
    z = (1/(2*np.pi*sigma_x*sigma_y) * np.exp(-((x-muX)**2/(2*sigma_x**2)
         + (y-muY)**2/(2*sigma_y**2))))    
    return z


def plotTri(tri, points):
    plt.triplot(points[:,0], points[:,1], tri.simplices)
    plt.plot(points[:,0], points[:,1], 'o')
    plt.show()
    
    
def generateOrderedGrid(xmin, xmax, ymin, ymax, kstep):
    x1 = np.arange(xmin, xmax, kstep)
    x1 = np.append(x1, x1[0]*-1)
    x2 = np.arange(ymin, ymax, kstep)
    x2 = np.append(x2, x2[0]*-1)
    x=[]
    y=[]
    X, Y = np.meshgrid(x1, x2)
    for i in range(len(x1)):
        for j in range(len(x2)):
            x.append(X[i,j])
            y.append(Y[i,j])       
    mesh = np.asarray([x,y]).T
    return mesh

def generateOrderedGridCenteredAtZero(xmin, xmax, ymin, ymax, kstep, includeOrigin = True):
    stepsX = int(np.ceil(np.ceil((abs(xmin) + abs(xmax)) / (kstep))/2))
    x =[]
    x.append(0)
    for i in range(1, stepsX):
        x.append(i*kstep)
        x.append(-i*kstep)
        
    stepsY = int(np.ceil(np.ceil((abs(ymin)+ abs(ymax)) / (kstep))/2))
    y =[]
    y.append(0)
    for i in range(1, stepsY):
        y.append(i*kstep)
        y.append(-i*kstep)

    X, Y = np.meshgrid(x, y)
    x1 = []
    x2 = []
    for i in range(len(x)):
        for j in range(len(y)):
            x1.append(X[i,j])
            x2.append(Y[i,j])       
   
    mesh = np.asarray([x2,x1]).T
    if includeOrigin == False:
        mesh = np.delete(mesh,0,0)
    return mesh
    
