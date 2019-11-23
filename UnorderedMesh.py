# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 17:28:43 2019

@author: Ryleigh
"""
import numpy as np
import XGrid
import random
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import Functions as fun
import Operations2D
# xCoord, yCoord, is the point we are looking for the closests 
# two points to. 
# AllPoints is a Nx2  matrix of all the degrees of freedom.
#  x1   y1
#  x2   y2
#  x3   y3
#  ...  ...

def findNearestThreePoints(xCoord, yCoord, AllPoints):
    normList = []
    for point in range(len(AllPoints)):
        xVal = AllPoints[point,0]
        yVal = AllPoints[point,1]
        normList.append(np.sqrt((xCoord-xVal)**2+(yCoord-yVal)**2))
    idx = np.argsort(normList)
    print(idx)
    if idx[0] == 0: # point is part of the set
        return np.asarray([[AllPoints[idx[1],0], AllPoints[idx[1],1]], [AllPoints[idx[2],0], AllPoints[idx[2],1]], [AllPoints[idx[3],0], AllPoints[idx[3],1]]])
    else:
        return np.asarray([[AllPoints[idx[0],0], AllPoints[idx[0],1]], [AllPoints[idx[1],0], AllPoints[idx[1],1]], [AllPoints[idx[2],0], AllPoints[idx[2],1]]])



# point: Point to center grid around
# spacing: step size of grid
# span: Units around the point to make the grid.
def makeOrderedGridAroundPoint(point, spacing, span, xmin, xmax, ymin, ymax):
    percent = 0.1
    x = XGrid.getCenteredXvecAroundPoint(spacing, span, span, point[0])
    y = XGrid.getCenteredXvecAroundPoint(spacing, span, span, point[1])
    x = np.asarray(x)
    y = np.asarray(y)
    x = x[(x>=xmin+percent*(xmax-xmin)) & (x<=xmax-percent*(xmax-xmin))]
    y = y[(y>=ymin+percent*(xmax-xmin)) & (y<=ymax-percent*(xmax-xmin))]
    if (min(x)<xmin) | (max(x)>xmax)| (min(y)<ymin) | (max(y)>ymax):
        print('Problem in making grid')
    X, Y = np.meshgrid(x, y)
    xVec = np.reshape(X,[1,np.size(X)],1).T
    yVec = np.reshape(Y,[1,np.size(Y)],1).T
    points = np.hstack((xVec,yVec))
    return points



#(Px,Py): point we want to approximate
# nearestPoints 
#  x1   y1
#  x2   y2
#  x3   y3
def baryInterp(Px, Py, simplexPoints, degsFreePDF):
    Xv1 = simplexPoints[0,0]
    Yv1 = simplexPoints[0,1]
    Xv2 = simplexPoints[1,0]
    Yv2 = simplexPoints[1,1]
    Xv3 = simplexPoints[2,0]
    Yv3 = simplexPoints[2,1]
#    PDF1 = np.log(degsFreePDF[0])
#    PDF2 = np.log(degsFreePDF[1])
#    PDF3 = np.log(degsFreePDF[2])
    PDF1 = degsFreePDF[0]
    PDF2 = degsFreePDF[1]
    PDF3 = degsFreePDF[2]
    Wv1 = ((Yv2-Yv3)*(Px-Xv3)+(Xv3-Xv2)*(Py-Yv3))/((Yv2-Yv3)*(Xv1-Xv3)+(Xv3-Xv2)*(Yv1-Yv3))
    Wv2 = ((Yv3-Yv1)*(Px-Xv3)+(Xv1-Xv3)*(Py-Yv3))/((Yv2-Yv3)*(Xv1-Xv3)+(Xv3-Xv2)*(Yv1-Yv3))
    Wv3 = 1-Wv1-Wv2
    if (Wv1 < -10**(-10)) | (Wv2 < -10**(-10)) | (Wv3 < -10**(-10)):
        #print(min(Wv1,Wv2,Wv3))
        return 0
#        plt.figure()
#        plt.plot(simplexPoints[0,0], simplexPoints[0,1], '*k')
#        plt.plot(simplexPoints[1,0], simplexPoints[1,1], '*k')
#        plt.plot(simplexPoints[2,0], simplexPoints[2,1], '*k')
#        plt.plot(Px, Py, '.r')
#        plt.show()
#    assert Wv1 >= 0, 'Weight less than 0'
#    assert Wv2 >= 0, 'Weight less than 0'
#    assert Wv3 >= 0, 'Weight less than 0'

    PDFNew = Wv1*PDF1+Wv2*PDF2+Wv3*PDF3
#    PDF = np.exp(PDFNew)

    return PDFNew

#x = baryInterp(-0.25,-0.5, np.asarray([[-0.25,-0.5], [0,1], [1,1]]),[1,1,2])

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


#def generateICGrid(x1, x2, init, h):
#    w1 = Operations2D.find_nearest(x1, 0)
#    w2 = Operations2D.find_nearest(x2, 0)
#    phat = np.zeros([len(x1), len(x2)])
#    a1 = init + fun.f1(init,0)
#    b1 = np.abs(fun.g1() * np.sqrt(h))
#    a2 = init + fun.f2(init,0)
#    b2 = np.abs(fun.g2() * np.sqrt(h))
#    phat0 = fun.dnorm(x1, a1, b1)  # pdf after one time step with Dirac \delta(x-init)
#    phat1 = fun.dnorm(x2, a2, b2)  # pdf after one time step with Dirac \delta(x-init)
#    phat[w1, :] = phat1
#    phat[:, w2] = phat0
#    PDF = np.reshape(phat,[1,np.size(phat)],1).T
#    
#    X, Y = np.meshgrid(x1,x2)
#    xVec = np.reshape(X,[1,np.size(X)],1).T
#    yVec = np.reshape(Y,[1,np.size(Y)],1).T
#    points = np.hstack((xVec,yVec))
#    
#    return points, PDF
    
def generateICPDF(x,y,sigma_x, sigma_y):
    z = (1/(2*np.pi*sigma_x*sigma_y) * np.exp(-(x**2/(2*sigma_x**2)
         + y**2/(2*sigma_y**2))))

#    fig = plt.figure()
#    ax = Axes3D(fig)
#    ax.scatter(x, y, z, c='r', marker='.')
    
    return z

# Function that finds the vertices for 
    # calculation of the barycentric interpolation.
def getVerticesForPoint(point, allPoints, tri):
    simplex = tri.find_simplex(point)
    verts = tri.simplices[simplex]
    vertices = []
    vertices.append(allPoints[verts[0]])
    vertices.append(allPoints[verts[1]])
    vertices.append(allPoints[verts[2]])
    vertices = np.asarray(vertices)
#    plt.figure()
#    plt.plot(vertices[0,0], vertices[0,1], '*k')
#    plt.plot(vertices[1,0], vertices[1,1], '*k')
#    plt.plot(vertices[2,0], vertices[2,1], '*k')
#    plt.plot(point[0], point[1], '.r')
#    plt.show()
    return np.asarray(vertices), verts

    
def getPDFForPoint(PDF, verts):
    newPDF = []
    newPDF.append(PDF[verts[0]])
    newPDF.append(PDF[verts[1]])
    newPDF.append(PDF[verts[2]])
    
    return np.asarray(newPDF)



def plotTri(tri, points):
    plt.triplot(points[:,0], points[:,1], tri.simplices)
    plt.plot(points[:,0], points[:,1], 'o')
    plt.show()
    
    
def generateOrderedGrid(xmin, xmax, ymin, ymax, kstep):
    x1 = np.arange(xmin, xmax, kstep)
    x2 = np.arange(ymin, ymax, kstep)
    x=[]
    y=[]
    X, Y = np.meshgrid(x1, x2)
    for i in range(len(x1)):
        for j in range(len(x2)):
            x.append(X[i,j])
            y.append(Y[i,j])       
    mesh = np.asarray([x,y]).T
    return mesh

#points = generateRandomPoints(-1,1,-1,1,1000)
#tri = Delaunay(points)
#vertices = getVerticesForPoint([0,0], points, tri)
#
#
#plotTri(tri, points)
