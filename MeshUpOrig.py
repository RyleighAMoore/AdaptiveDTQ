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
import GenerateLejaPoints as LP
import distanceMetrics
import LejaPointsToRemove as LPR

global MaxSlope
MaxSlope = 0 # Initialize to 0, the real value is set in the code
addPointsToBoundaryIfBiggerThanTolerance = 10**(-3)
removeZerosValuesIfLessThanTolerance = 10**(-10)
minDistanceBetweenPoints = 0.03
minDistanceBetweenPointsBoundary = 0.1
skipCount = 4
maxDistanceBetweenPoints = 0.15
numStdDev = 5 #For grids around each point in the mesh

adjustDensity = True
adjustBoundary = True

# addPointsToBoundaryIfBiggerThanTolerance = 10**(-2)
# removeZerosValuesIfLessThanTolerance = 10**(-20)
# minDistanceBetweenPoints = 0.1
# minDistanceBetweenPointsBoundary = 0.1
# skipCount = 5
# maxDistanceBetweenPoints = 0.2


def addPointsToMeshProcedure(Mesh, GMat, Grids, Vertices, VerticesNum, Pdf, triangulation, kstep, h, xmin, xmax, ymin, ymax):
    changedBool2 = 0
    changedBool1 = 0
    if adjustDensity:
        Mesh, GMat, Grids, Vertices, VerticesNum, Pdf, triangulation, changedBool2 = addInteriorPoints(Mesh, GMat, Grids, Vertices, VerticesNum, Pdf, triangulation, kstep, h)
    if adjustBoundary:
        Mesh, GMat, Grids, Vertices, VerticesNum, Pdf, triangulation, changedBool1 = addPointsToBoundary(Mesh, GMat, Grids, Vertices, VerticesNum, Pdf, triangulation, kstep, h)
    ChangedBool = max(changedBool1, changedBool2)
    return Mesh, GMat, Grids, Vertices, VerticesNum, Pdf, triangulation, ChangedBool, xmin, xmax, ymin, ymax 

def removePointsFromMeshProcedure(GMat, Mesh, Grids, Pdf, tri, boundaryOnlyBool):
    '''Removes the flagged values from the list of mesh values and in Gmat. 
    boolZerosArray is the list of zeros and ones denoting which grid points to remove.
    Gmat, Mesh, Grids, Vertices, and VerticesNum are all used in the 2DTQ-UnorderedMesh method 
    and the parts associated with the removed points need to be removed.'''
    ChangedBool2 = 0
    ChangedBool1 = 0
    if adjustBoundary:
        GMat, Mesh, Grids, Pdf, ChangedBool2 = removeBoundaryPoints(GMat, Mesh, Grids, Pdf, tri, boundaryOnlyBool)
    if adjustDensity:
        GMat, Mesh, Grids, Pdf, ChangedBool1 = removeInteriorPointsToMakeLessDense(GMat, Mesh, Grids, Pdf, tri, boundaryOnlyBool)
    ChangedBool = max(ChangedBool1, ChangedBool2)
    return GMat, Mesh, Grids, Pdf, ChangedBool


def getBoundaryPoints(Mesh, tri, alpha):
    edges = alpha_shape(Mesh, tri, alpha, only_outer=True)
    aa = list(chain(edges))
    out = [item for t in aa for item in t]
    pointsOnEdge = np.sort(out)
    pointsOnEdge = pointsOnEdge[1::2]  # Skip every other element to remove repeated elements
    return pointsOnEdge


def checkIntegrandForZeroPoints(GMat, PDF, tolerance, Mesh, tri, boundaryOnly):
    '''Check if the integrand p*G is less than the tolerance.
    Uses alpha hull to get the boundary points if boundaryOnly is True.'''
    maxMat = 10*np.ones(len(PDF))
    if boundaryOnly:
        pointsOnEdge = getBoundaryPoints(Mesh, tri, maxDistanceBetweenPoints)
        for i in pointsOnEdge:
            maxMat[i]=(np.max(PDF[i]*GMat[i]))    
    else:
        for i in range(len(PDF)):
            maxMat[i]=(np.max(PDF[i]*GMat[i]))  
    Slopes = getSlopes(Mesh, PDF)
    adding = np.asarray([Slopes < 0.001])
    adding2 = [np.asarray(maxMat) < tolerance]
    # print("#############")
    # print(np.sum(np.asarray(adding)))
    # print(np.sum(np.asarray(adding2)))
    possibleZeros = np.logical_and(adding, adding2)
    # print(np.sum(possibleZeros))
    # print("#############")

    return np.asarray(possibleZeros).T


def checkIntegrandForAddingPointsAroundBoundaryPoints(GMat, PDF, tolerance, Mesh, tri, boundaryOnly):
    maxMat = -1*np.ones(len(PDF))
    if boundaryOnly:
        pointsOnEdge = getBoundaryPoints(Mesh, tri, maxDistanceBetweenPoints)
        for i in pointsOnEdge:
            maxMat[i]=(np.max(PDF[i]*GMat[i]))    
    else:
        for i in range(len(PDF)):
            maxMat[i]=(np.max(PDF[i]*GMat[i]))
    addingAround = [np.asarray(maxMat) >= tolerance]
    # print(np.max(np.asarray(maxMat)))
    return np.asarray(addingAround).T

#def checkIntegralForZeroPoints(GMat, PDF, tolerance):
#    newPDF = []
#    for i in range(len(PDF)):
#        newPDF.append(np.matmul(np.asarray(GMat),PDF))
#    return [np.asarray(newPDF) <= tolerance]


def generateGRow(point, allPoints, kstep, h):
    row = []
    OrderA = []
    for i in range(len(allPoints)):
        val = kstep**2*fun.G(point[0], point[1], allPoints[i,0], allPoints[i,1], h)
        row.append(val)
        OrderA.append([point[0], point[1], allPoints[i,0], allPoints[i,1]])
    return row



def removeBoundaryPoints(GMat, Mesh, Grids, Pdf, tri, boundaryOnlyBool):
    stillRemoving = True
    ChangedBool = 0
    length = len(Mesh)
    while stillRemoving: # Removing boundary points
        boundaryZeroPointsBoolArray = checkIntegrandForZeroPoints(GMat, Pdf, removeZerosValuesIfLessThanTolerance,Mesh,tri, True)
        if max(boundaryZeroPointsBoolArray == 1):
            for val in range(len(boundaryZeroPointsBoolArray)-1,-1,-1):
                if boundaryZeroPointsBoolArray[val] == 1: # remove the point
                    ChangedBool=1
                    GMat, Mesh, Grids, Pdf = removePoint(val, GMat, Mesh, Grids, Pdf)
        else:
            stillRemoving = False
        Vertices, VerticesNum, tri = houseKeepingAfterAdjustingMesh(Mesh, Grids, tri)
    for i in range(len(Mesh)-1,-1,-1): # Remove straggling points
        nearestPoint, distToNearestPoints = UM.findNearestKPoints(Mesh[i,0],Mesh[i,1], Mesh, 6)            # print("Making Less Dense!...")
        dist = np.mean(distToNearestPoints)
        if dist > maxDistanceBetweenPoints: # Remove outlier
            GMat, Mesh, Grids, Pdf = removePoint(i, GMat, Mesh, Grids, Pdf)
            ChangedBool = 1
    if ChangedBool == 1:
        Vertices, VerticesNum, tri = houseKeepingAfterAdjustingMesh(Mesh, Grids, tri)
    print("Boundary points removed", length -len(Mesh))  
    return GMat, Mesh, Grids, Pdf, ChangedBool


def removePoint(index, GMat, Mesh, Grids, Pdf):
    GMat.pop(index)
    Mesh = np.delete(Mesh, index, 0)
    Grids.pop(index)
    Pdf = np.delete(Pdf, index, 0)
    return GMat, Mesh, Grids, Pdf
    

def removeInteriorPointsToMakeLessDense(GMat, Mesh, Grids, Pdf, tri, boundaryOnlyBool):
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
    if spacing < maxDistanceBetweenPoints*(skipCount+1)/skipCount: # if removing points will be ok.
        indices = LPR.getMeshIndicesToRemoveFromMesh(meshWithSmallSlopes, skipCount)
        for j in range(len(indices)-1,-1,-1): # Check if point is likely top of hill - don't remove it
            nearestPoint, distances = UM.findNearestKPoints(Mesh[corrIndices[j],0],Mesh[corrIndices[j],1], meshWithSmallSlopes, 6)            # print("Making Less Dense!...")
            distToNearestPoint = np.max(distances)
            if distToNearestPoint < maxDistanceBetweenPoints:
                GMat, Mesh, Grids, Pdf = removePoint(corrIndices[j], GMat, Mesh, Grids, Pdf)
                ChangedBool = 1
            # else:
            #     print("Skip removing top of hill")
    else:
        print("\nSkipping making less dense, the current spacing ", spacing, ">", maxDistanceBetweenPoints*(skipCount+1)/skipCount)
    numReduced = startingLength-len(Mesh)
    print("Removed ", numReduced, "to decrease density.")
    if ChangedBool:
        Vertices, VerticesNum, tri = houseKeepingAfterAdjustingMesh(Mesh, Grids, tri)
    return GMat, Mesh, Grids, Pdf, ChangedBool


def houseKeepingAfterAdjustingMesh(Mesh, Grids, tri):
    '''Updates all the Vertices information for the mesh. Must be run after removing points'''
    tri = Delaunay(Mesh, incremental=True)
    Vertices = []
    VerticesNum = []
    for point in range(len(Mesh)): # Recompute Vertices and VerticesNum matrices
        grid = Grids[point]
        Vertices.append([])
        VerticesNum.append([])
        for currGridPoint in range(len(grid)):
            vertices, indices = UM.getVerticesForPoint([grid[currGridPoint,0], grid[currGridPoint,1]], Mesh, tri) # Points that make up triangle
            Vertices[point].append(np.copy(vertices))
            VerticesNum[point].append(np.copy(indices))
    return Vertices, VerticesNum, tri

import untitled7 as u7
from scipy.interpolate import griddata
def addPoint(Px,Py, Mesh, GMat, Grids, Vertices, VerticesNum, Pdf, triangulation, kstep, h):
    Mesh = np.append(Mesh, np.asarray([[Px],[Py]]).T, axis=0)
    xmin = np.min(Mesh[:,0]); xmax = np.max(Mesh[:,0])
    ymin = np.min(Mesh[:,1]); ymax = np.max(Mesh[:,1])
    grid = UM.makeOrderedGridAroundPoint([Px,Py],kstep, max(xmax-xmin, ymax-ymin),Px-numStdDev*np.sqrt(h)*fun.g1() ,Px+numStdDev*np.sqrt(h)*fun.g1(),Py-numStdDev*np.sqrt(h)*fun.g2(),Py+numStdDev*np.sqrt(h)*fun.g2())
    Grids.append(np.copy(grid))
    Vertices.append([])
    VerticesNum.append([])
    for currGridPoint in range(len(grid)):
        vertices, indices = UM.getVerticesForPoint([grid[currGridPoint,0], grid[currGridPoint,1]], Mesh, triangulation) # Points that make up triangle
        Vertices[-1].append(np.copy(vertices))
        VerticesNum[-1].append(np.copy(indices))
        
    pointVertices, pointIndices = UM.getVerticesForPoint([Px,Py], Mesh, triangulation) # Points that make up triangle    
    threePdfVals = [Pdf[pointIndices[0]], Pdf[pointIndices[1]], Pdf[pointIndices[2]]]
    
    train_samples, train_values = u7.getMeshValsThatAreClose(Mesh, Pdf, 0.1, 0.1, Px, Py)
    grid_z2 = griddata(train_samples, train_values, np.asarray([[Px,Py]]), method='cubic', fill_value=0)
    
    interp = UM.baryInterp([Px],[Py], pointVertices, threePdfVals)
    try: 
        threePdfVals = [Pdf[pointIndices[0]], Pdf[pointIndices[1]], Pdf[pointIndices[2]]]
        interp = UM.baryInterp([Px],[Py], pointVertices, threePdfVals)
        Pdf = np.append(Pdf, interp, axis=0)                      
    except:
        interp = 0
        Pdf = np.append(Pdf, [10**(-20)], axis=0)
    # print(interp)
    gRow = generateGRow([Px, Py], grid, kstep, h)
    GMat.append(np.copy(gRow))
    triangulation.add_points(np.asarray([[Px],[Py]]).T, restart=False)
    # if interp > 0.1:
    #     plt.figure()
    #     plt.plot(Mesh[:,0], Mesh[:,1], '.k')
    #     plt.plot(Px,Py, '*r')
    #     plt.show()
    return  Mesh, GMat, Grids, Vertices, VerticesNum, Pdf, triangulation
 
    
def addPointsToBoundary(Mesh, GMat, Grids, Vertices, VerticesNum, Pdf, triangulation, kstep, h):
    numBoundaryAdded = 0
    keepAdding = True
    changedBool = 0
    print("adding boundary points...")
    while keepAdding:
        boundaryPointsToAddAround = checkIntegrandForAddingPointsAroundBoundaryPoints(GMat, Pdf, addPointsToBoundaryIfBiggerThanTolerance, Mesh, triangulation, True)
        # print(np.count_nonzero(boundaryPointsToAddAround))
        # plt.figure()
        # pointsX = []
        # pointsY = []
        # for i in range(len(boundaryPointsToAddAround)):
        #     if boundaryPointsToAddAround[i]==1:
        #         pointsX.append(Mesh[i,0])
        #         pointsY.append(Mesh[i,1])
        # plt.plot(np.asarray(pointsX), np.asarray(pointsY), '.')
        # plt.show()
        
        if max(boundaryPointsToAddAround == 1) and np.count_nonzero(boundaryPointsToAddAround) > 5:
            for val in range(len(boundaryPointsToAddAround)-1,-1,-1):
                if boundaryPointsToAddAround[val] == 1: # if we should extend boundary
                    # newPoints = addPointsRadially(Mesh[val,0], Mesh[val,1], Mesh, 4, maxDistanceBetweenPoints/2, 0.0001)
                    allPoints, newPoints = LP.getLejaPointsWithStartingPoints(Mesh[val,0], Mesh[val,1], 3, Mesh, 3, np.sqrt(h)*fun.g1(),np.sqrt(h)*fun.g2(),6, 100)
                    newPoints = checkIfDistToClosestPointIsOk(newPoints, Mesh, minDistanceBetweenPointsBoundary)
                    for point in range(len(newPoints)):
                        ChangedBool = 1
                        Mesh, GMat, Grids, Vertices, VerticesNum, Pdf, triangulation = addPoint(newPoints[point,0], newPoints[point,1], Mesh, GMat, Grids, Vertices, VerticesNum, Pdf, triangulation, kstep, h)
                        numBoundaryAdded = numBoundaryAdded +1
        else:
            keepAdding =False
        # plt.figure()
        # plt.plot(Mesh[:,0], Mesh[:,1], '.k')
        # plt.show()
        if changedBool == 1:
            Vertices, VerticesNum, tri = houseKeepingAfterAdjustingMesh(Mesh, Grids, triangulation)
    print("# boundary points Added = ", numBoundaryAdded)    
    return Mesh, GMat, Grids, Vertices, VerticesNum, Pdf, triangulation, changedBool

def addInteriorPoints(Mesh, GMat, Grids, Vertices, VerticesNum, Pdf, triangulation, kstep, h):
    Slopes = getSlopes(Mesh, Pdf)
    denisfyAroundPointIfSlopeLargerThanTolerance = 0.1 # np.quantile(Slopes,0.5)
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
    #                newPoints = addPointsRadially(Mesh[val,0], Mesh[val,1], Mesh, 4, kstep/2) 
                    allPoints, newPoints = LP.getLejaPointsWithStartingPoints(Mesh[val,0], Mesh[val,1], 4, Mesh, 4, np.sqrt(h)*fun.g1(),np.sqrt(h)*fun.g2(), 6,100)
                    newPoints = checkIfDistToClosestPointIsOk(newPoints, Mesh, min(minDistanceBetweenPoints/Slopes[val], minDistanceBetweenPoints))
                    for point in range(len(newPoints)):
                        ChangedBool = 1
                        Mesh, GMat, Grids, Vertices, VerticesNum, Pdf, triangulation = addPoint(newPoints[point,0], newPoints[point,1], Mesh, GMat, Grids, Vertices, VerticesNum, Pdf, triangulation, kstep, h)
        print("# interior points Added = ", numInteriorAdded)  
        return Mesh, GMat, Grids, Vertices, VerticesNum, Pdf, triangulation, ChangedBool 


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


def addPointsRadially(pointX, pointY, mesh, numPointsToAdd, kstep, minDist):
    #radius = DM.fillDistance(mesh)
    radius = kstep
    dTheta = 2*np.pi/numPointsToAdd
    points = []
    for i in range(numPointsToAdd):
        newPointX = radius*np.cos(i*dTheta)+pointX
        newPointY = radius*np.sin(i*dTheta) + pointY
        nearestPoint = UM.findNearestPoint(newPointX, newPointY, mesh)
        distToNearestPoint = np.sqrt((nearestPoint[0,0] - newPointX)**2 + (nearestPoint[0,1] - newPointY)**2)
        if distToNearestPoint > minDist:
            #print("adding")
            points.append([newPointX, newPointY])
    return np.asarray(points)
    
#points = addPointsRadially(1,-2,mesh, 50)   
#plt.plot(points[:,0], points[:,1], '.')

def checkIfDistToClosestPointIsOk(newPoints, Mesh, minDist):
    '''Checks to make sure that a new point we want to add is not too close or too far from another points'''
    points = []
    for i in range(len(newPoints)):
        newPointX = newPoints[i,0]
        newPointY = newPoints[i,1]
        nearestPoint = UM.findNearestPoint(newPointX, newPointY, Mesh)
        distToNearestPoint = np.sqrt((nearestPoint[0,0] - newPointX)**2 + (nearestPoint[0,1] - newPointY)**2)
        if distToNearestPoint > minDist and distToNearestPoint < maxDistanceBetweenPoints:
            points.append([newPointX, newPointY])
        # else:
        #     print("Nope")
    return np.asarray(points)

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
