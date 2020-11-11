# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 10:38:41 2019

@author: Ryleigh
"""

import numpy as np
import Functions as fun
import UnorderedMesh as UM
from scipy.spatial import Delaunay
import distanceMetrics as DM
from itertools import chain
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyopoly1 import LejaPoints as LP
import distanceMetrics 
from pyopoly1 import LejaPointsToRemove as LPR
from pyopoly1 import LejaPoints as LP
from scipy.interpolate import griddata


''''Tolerance Parameters'''
addPointsToBoundaryIfBiggerThanTolerance = 10**(-3)
removeZerosValuesIfLessThanTolerance = 10**(-4)
minDistanceBetweenPoints = 0.15
minDistanceBetweenPointsBoundary = 0.15
maxDistanceBetweenPoints = 0.15



def addPointsToMeshProcedure(Mesh, Pdf, triangulation, kstep, h, poly, GMat, adjustBoundary =True, adjustDensity=False):
    '''If the mesh is changed, these become 1 so we know to recompute the triangulation'''
    changedBool2 = 0 
    changedBool1 = 0
    meshSize = len(Mesh)
    if adjustDensity:
        Mesh, Pdf, triangulation, changedBool2 = addInteriorPoints(Mesh, Pdf, triangulation)
    if adjustBoundary:
        Mesh, Pdf, triangulation, changedBool1 = addPointsToBoundary(Mesh, Pdf, triangulation)
    ChangedBool = max(changedBool1, changedBool2)
    if ChangedBool==1:
        newMeshSize = len(Mesh)
        for i in range(meshSize+1, newMeshSize+1):
            GMat = fun.AddPointToG(Mesh[:i,:], i-1, h, GMat)
    return Mesh, Pdf, triangulation, ChangedBool, GMat

def removePointsFromMeshProcedure(Mesh, Pdf, tri, boundaryOnlyBool, poly, GMat, LPMat, LPMatBool, adjustBoundary =True, adjustDensity=False):
    '''If the mesh is changed, these become 1 so we know to recompute the triangulation'''
    ChangedBool2 = 0
    ChangedBool1 = 0
    ChangedBool3 = 0
    if adjustBoundary:
        Mesh, Pdf, ChangedBool2, GMat,LPMat, LPMatBool = removeBoundaryPoints(Mesh, Pdf, tri, boundaryOnlyBool, GMat, LPMat, LPMatBool)
    # if adjustDensity:
    #     Mesh, Pdf, ChangedBool1 = removeInteriorPointsToMakeLessDense(Mesh, Pdf, tri, boundaryOnlyBool, poly)
    Mesh, Pdf, ChangedBool3, GMat = checkForAndRemoveZeroPoints(Mesh,Pdf, tri, GMat)
    ChangedBool = max(ChangedBool1, ChangedBool2, ChangedBool3)
    return Mesh, Pdf, ChangedBool, GMat, LPMat, LPMatBool

def checkForAndRemoveZeroPoints(Mesh, Pdf, tri, GMat):
    print("Checking for small points to remove....")
    ChangedBool=False
    if np.min(Pdf) < removeZerosValuesIfLessThanTolerance:
        zeros =  np.asarray([Pdf < removeZerosValuesIfLessThanTolerance])
        for i in range(len(zeros)):
            if zeros.T[i][0]==1:
                Mesh, Pdf,GMat = removePoint(i, Mesh, Pdf, GMat)
        print("Removed", (np.sum(zeros)), "points that were tiny")
        ChangedBool =True
    if ChangedBool == 1:
        tri = houseKeepingAfterAdjustingMesh(Mesh, tri)
    return Mesh, Pdf, ChangedBool, GMat
            

def getBoundaryPoints(Mesh, tri, alpha):
    '''Uses triangulation and alpha hull technique to find boundary points'''
    edges = alpha_shape(Mesh, tri, alpha, only_outer=True)
    aa = list(chain(edges))
    out = [item for t in aa for item in t]
    pointsOnBoundary = np.sort(out)
    pointsOnBoundary = pointsOnBoundary[1::2]  # Skip every other element to remove repeated elements
    return pointsOnBoundary


def checkIntegrandForRemovingSmallPoints(PDF, Mesh, tri):
    '''Check if any points are tiny and can be removed'''
    possibleZeros = [np.asarray(PDF) < removeZerosValuesIfLessThanTolerance] # want value to be small
    return np.asarray(possibleZeros).T


def checkIntegrandForAddingPointsAroundBoundaryPoints(PDF, addPointsToBoundaryIfBiggerThanTolerance, Mesh, tri):
    '''Check if the points on the edge are too big and we need more points around them
    Uses alpha hull to get the boundary points if boundaryOnly is True.'''
    valueList = -1*np.ones(len(PDF)) # Set to small values for placeholder
    pointsOnEdge = getBoundaryPoints(Mesh, tri, maxDistanceBetweenPoints*2)
    for i in pointsOnEdge:
        valueList[i]=PDF[i]
    addingAround = [np.asarray(valueList) >= addPointsToBoundaryIfBiggerThanTolerance]
    return np.asarray(addingAround).T


def removeBoundaryPoints(Mesh, Pdf, tri, boundaryOnlyBool, GMat, LPMat, LPMatBool):
    stillRemoving = True
    ChangedBool = 0
    initLength = len(Mesh)
    indexRem = []
    '''# Removing boundary points'''
    while stillRemoving: 
        boundaryZeroPointsBoolArray = checkIntegrandForRemovingSmallPoints(Pdf,Mesh,tri)
        if max(boundaryZeroPointsBoolArray == 1):
            for val in range(len(boundaryZeroPointsBoolArray)-1,-1,-1):
                if boundaryZeroPointsBoolArray[val] == 1: # remove the point
                    ChangedBool=1
                    Mesh, Pdf, GMat = removePoint(val, Mesh, Pdf, GMat)
                    indexRem.append(val)
        else: # Stop removing points
            stillRemoving = False
        tri = houseKeepingAfterAdjustingMesh(Mesh, tri)
        
        Mat = np.zeros(np.shape(LPMat))
        for ind in indexRem:
            larger = LPMat > ind
            Mat = Mat + larger
            LPMatBool[val] = False
        
    indexRem = []
    '''Remove straggling points'''
    for i in range(len(Mesh)-1,-1,-1): 
        nearestPoint, distToNearestPoints = UM.findNearestKPoints(Mesh[i,0],Mesh[i,1], Mesh, 6)
        dist = np.mean(distToNearestPoints)
        if dist > 2*maxDistanceBetweenPoints: # Remove outlier
            Mesh, Pdf, GMat = removePoint(i, Mesh, Pdf, GMat)
            ChangedBool = 1
            indexRem.append(i)
            print("outlier")
    
    if ChangedBool == 1:
        tri = houseKeepingAfterAdjustingMesh(Mesh, tri)
        
    for ind in indexRem:
            larger = LPMat > ind
            Mat = Mat + larger
            LPMatBool[val] = False
    
    LPMat = np.copy(LPMat) - Mat
    
    print("Boundary points removed", initLength -len(Mesh))  
    return Mesh, Pdf, ChangedBool, GMat, LPMat, LPMatBool


def removePoint(index, Mesh, Pdf, GMat):
    Mesh = np.delete(Mesh, index, 0)
    Pdf = np.delete(Pdf, index, 0)
    GMat = np.delete(GMat, index,0)
    GMat = np.delete(GMat, index,1)
    return Mesh, Pdf, GMat

def houseKeepingAfterAdjustingMesh(Mesh, tri):
    '''Updates all the Vertices information for the mesh. Must be run after removing points'''
    tri = Delaunay(Mesh, incremental=True)
    return tri


def addPoint(Px,Py, Mesh, Pdf, triangulation):
    newPoint = np.asarray([[Px],[Py]]).T
    interp = np.asarray([griddata(Mesh, Pdf, newPoint, method='cubic', fill_value=np.min(Pdf))])
    if interp < 0:
        interp = np.asarray([griddata(Mesh, Pdf, newPoint, method='linear', fill_value=np.min(Pdf))])
    if interp <=0:
        interp = np.asarray([[10**(-10)]])

    Mesh = np.append(Mesh, np.asarray([[Px],[Py]]).T, axis=0)
    Pdf = np.append(Pdf, interp[0], axis=0)                      
    triangulation.add_points(np.asarray([[Px],[Py]]).T, restart=False)
    return  Mesh, Pdf, triangulation

    
def addPointsToBoundary(Mesh, Pdf, triangulation):
    numBoundaryAdded = 0
    keepAdding = True
    ChangedBool = 0
    print("adding boundary points...")
    count = 0
    while keepAdding and count < 3:
        boundaryPointsToAddAround = checkIntegrandForAddingPointsAroundBoundaryPoints(Pdf, addPointsToBoundaryIfBiggerThanTolerance, Mesh, triangulation)
        if max(boundaryPointsToAddAround == 1):
            for val in range(len(boundaryPointsToAddAround)-1,-1,-1):
                if boundaryPointsToAddAround[val] == 1: # if we should extend boundary
                    newPoints = addPointsRadially(Mesh[val,0], Mesh[val,1], Mesh, 8)
                    newPoints = checkIfDistToClosestPointIsOk(newPoints, Mesh)
                    # if len(newPoints)==0:
                    #     keepAdding = False
                    for point in range(len(newPoints)):
                        ChangedBool = 1
                        Mesh, Pdf, triangulation = addPoint(newPoints[point,0], newPoints[point,1], Mesh, Pdf, triangulation)
                        numBoundaryAdded = numBoundaryAdded + 1
        else:
            keepAdding =False
        if ChangedBool == 1:
            tri = houseKeepingAfterAdjustingMesh(Mesh, triangulation)
        count = count+1
    print("# boundary points Added = ", numBoundaryAdded)    
    return Mesh, Pdf, triangulation, ChangedBool


def addPointsRadially(pointX, pointY, mesh, numPointsToAdd):
    radius = minDistanceBetweenPointsBoundary
    dTheta = 2*np.pi/numPointsToAdd
    points = []
    for i in range(numPointsToAdd):
        newPointX = radius*np.cos(i*dTheta)+pointX
        newPointY = radius*np.sin(i*dTheta) + pointY
        nearestPoint,idx = UM.findNearestPoint(newPointX, newPointY, mesh)
        distToNearestPoint= np.sqrt((nearestPoint[0,0] - newPointX)**2 + (nearestPoint[0,1] - newPointY)**2)
        if distToNearestPoint >= minDistanceBetweenPointsBoundary*0.9:
            points.append([newPointX, newPointY])
    return np.asarray(points)
    

def checkIfDistToClosestPointIsOk(newPoints, Mesh):
    '''Checks to make sure that a new point we want to add is not too close or too far from another points'''
    points = []
    for i in range(len(newPoints)):
        newPointX = newPoints[i,0]
        newPointY = newPoints[i,1]
        nearestPoint, dist = UM.findNearestPoint(newPointX, newPointY, Mesh, samePointRet0= True)
        distToNearestPoint = np.sqrt((nearestPoint[0,0] - newPointX)**2 + (nearestPoint[0,1] - newPointY)**2)
        if distToNearestPoint > minDistanceBetweenPoints*0.9 and distToNearestPoint < maxDistanceBetweenPoints*1.1:
            points.append([newPointX, newPointY])

    return np.asarray(points)

# https://stackoverflow.com/questions/23073170/calculate-bounding-polygon-of-alpha-shape-from-the-delaunay-triangulation
def alpha_shape(points, triangulation, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"
    
    def add_edge(edges, i, j):
        """
        Add an edge between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = triangulation
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        val = s * (s - a) * (s - b) * (s - c)
        if val <=0:
            circum_r = float('nan')
        else:
            area = np.sqrt(s * (s - a) * (s - b) * (s - c))
            circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
            
    # plt.figure()
    # plt.plot(points[:, 0], points[:, 1], '.')
    # for i, j in edges:
    #     plt.plot(points[[i, j], 0], points[[i, j], 1], 'r')
    # plt.show()
    return edges


'''Functions below here are still under construction
##################################################################'''

'''Used when using Leja like procedure to make mesh less dense'''
skipCount = 5 
global MaxSlope
MaxSlope = 0 # Initialize to 0, the real value is set in the code

def getSlopes(mesh, PDF):
    Slopes = []
    for i in range(len(mesh)):
        Px = mesh[i,0]
        Py = mesh[i,1]
        pdf1 = PDF[i]
        nearestPoint, index = UM.findNearestPoint(Px,Py, mesh, True)
        pdf2 = PDF[index]
        slope = (pdf1-pdf2)/(np.sqrt((Px-nearestPoint[0,0])**2 + (Py-nearestPoint[0,1])**2))
        Slopes.append(np.abs(slope))
    global MaxSlope
    if MaxSlope == 0: # set first
        MaxSlope = np.max(Slopes)
    return Slopes/np.max(Slopes)


def removeInteriorPointsToMakeLessDense(Mesh, Pdf, tri, boundaryOnlyBool, poly):
    ChangedBool=0
    startingLength = len(Mesh)
    Slopes = getSlopes(Mesh, Pdf)
    removePointsIfSlopeLessThanTolerance = 0.1 # np.quantile(Slopes,.3)
    pointsToRemove = np.asarray([np.asarray(Slopes) < removePointsIfSlopeLessThanTolerance]).T
    meshWithSmallSlopes = []
    corrIndices = [] # Indices in the bigger mesh for removal
    for i in range(len(Mesh)): # Assign indices for removal
        if pointsToRemove[i]==1:
            corrIndices.append(i) #Index from bigger mesh
            meshWithSmallSlopes.append(Mesh[i,:])
    meshWithSmallSlopes = np.asarray(meshWithSmallSlopes)
    
    corrIndices = np.sort(corrIndices)
    spacing = distanceMetrics.fillDistance(meshWithSmallSlopes)
    if spacing < 1: #maxDistanceBetweenPoints*(skipCount+1)/skipCount: # if removing points will be ok.
        indices = LPR.getMeshIndicesToRemoveFromMesh(meshWithSmallSlopes, skipCount, poly)
        for j in range(len(indices)-1,-1,-1): # Check if point is likely top of hill - don't remove it
            nearestPoint, distances = UM.findNearestKPoints(Mesh[corrIndices[j],0],Mesh[corrIndices[j],1], meshWithSmallSlopes, 6)            # print("Making Less Dense!...")
            distToNearestPoint = np.max(distances)
            if distToNearestPoint < maxDistanceBetweenPoints:
                Mesh, Pdf = removePoint(corrIndices[j], Mesh, Pdf)
                ChangedBool = 1
            # else:
            #     print("Skip removing top of hill")
    else:
        print("\nSkipping making less dense, the current spacing ", spacing, ">", maxDistanceBetweenPoints*(skipCount+1)/skipCount)
    numReduced = startingLength-len(Mesh)
    print("Removed ", numReduced, "to decrease density.")
    if ChangedBool:
        tri = houseKeepingAfterAdjustingMesh(Mesh, tri)
        
    # plt.scatter(Mesh[:,0], Mesh[:,1], c='r')

    return Mesh, Pdf, ChangedBool

def addInteriorPoints(Mesh, Pdf, triangulation):
    ChangedBool = 0
    numInteriorAdded = 0
    Slopes = getSlopes(Mesh, Pdf)
    denisfyAroundPointIfSlopeLargerThanTolerance =0.05 #np.quantile(Slopes,0.5) # 0.05
    interiorPointsToAddAround = np.asarray([np.asarray(Slopes)> denisfyAroundPointIfSlopeLargerThanTolerance]).T
    meshWithBigSlopes = []
    indexMax = np.argmax(Pdf) # We want to add around the largest value
    interiorPointsToAddAround[indexMax]=1
    for i in range(len(Mesh)):
        if interiorPointsToAddAround[i]==1:
            meshWithBigSlopes.append(Mesh[i,:])
    meshWithBigSlopes = np.asarray(meshWithBigSlopes)
    if len(meshWithBigSlopes) > 1:
        spacing = distanceMetrics.fillDistance(meshWithBigSlopes)
        ChangedBool = 0
        numInteriorAdded = 0
        if max(interiorPointsToAddAround == 1): 
            print("adding interior points...")
            for val in range(len(Slopes)-1,-1,-1):
                if (spacing > minDistanceBetweenPoints) and (interiorPointsToAddAround[val] == 1): # if we should extend boundary
                    newPoints = addPointsRadially(Mesh[val,0], Mesh[val,1], Mesh, 4,minDistanceBetweenPoints, minDistanceBetweenPoints/2) 
                    
                    newPoints = checkIfDistToClosestPointIsOk(newPoints, Mesh, min(minDistanceBetweenPoints/Slopes[val], minDistanceBetweenPoints))
                    for point in range(len(newPoints)):
                        ChangedBool = 1
                        numInteriorAdded+=1
                        Mesh, Pdf, triangulation = addPoint(newPoints[point,0], newPoints[point,1], Mesh, Pdf, triangulation)
        print("# interior points Added = ", numInteriorAdded)  
    return Mesh, Pdf, triangulation, ChangedBool 


def setGlobalVarsForMesh(mesh):
    ''''May be used in the future for making code more adaptable to different IC'''
    global minDistanceBetweenPoints
    global minDistanceBetweenPointsBoundary
    global maxDistanceBetweenPoints
    minDistanceBetweenPoints = distanceMetrics.separationDistance(mesh)*1.5
    minDistanceBetweenPointsBoundary = distanceMetrics.separationDistance(mesh)*1.5
    maxDistanceBetweenPoints = 1.5*minDistanceBetweenPoints

