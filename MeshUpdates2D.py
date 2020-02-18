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

#addPointsToBoundaryIfBiggerThanTolerance = 10**(-5)
#removeZerosValuesIfLessThanTolerance = 10**(-30)
##removePointsIfSlopeLessThanTolerance = 5
##denisfyAroundPointIfSlopeLargerThanTolerance = 5
#minDistanceBetweenPoints = 0.1

addPointsToBoundaryIfBiggerThanTolerance = 10**(-2)
removeZerosValuesIfLessThanTolerance = 10**(-10)
#removePointsIfSlopeLessThanTolerance = 5
#denisfyAroundPointIfSlopeLargerThanTolerance = 5
minDistanceBetweenPoints = 0.03
minDistanceBetweenPointsBoundary = 0.1
skipCount = 7
maxDistanceBetweenPoints = 0.15


def checkIntegrandForZeroPoints(GMat, PDF, tolerance, Mesh, tri, boundaryOnly):
    maxMat = 10*np.ones(len(PDF))
    if boundaryOnly:
        edges = alpha_shape(Mesh, tri, 0.2, only_outer=True)
        aa = list(chain(edges))
        out = [item for t in aa for item in t]
        pointsOnEdge = np.sort(out)
        pointsOnEdge = pointsOnEdge[1::2]  # Skip every other element to remove repeated elements
        for i in pointsOnEdge:
            maxMat[i]=(np.max(PDF[i]*GMat[i]))    
    else:
        for i in range(len(PDF)):
            maxMat[i]=(np.max(PDF[i]*GMat[i]))    
    possibleZeros = [np.asarray(maxMat) < tolerance]
    return np.asarray(possibleZeros).T


def checkIntegrandForAddingPointsAroundBoundaryPoints(GMat, PDF, tolerance, Mesh, tri, boundaryOnly):
    maxMat = -1*np.ones(len(PDF))
    if boundaryOnly:
        edges = alpha_shape(Mesh, tri, 0.2, only_outer=True)
        aa = list(chain(edges))
        out = [item for t in aa for item in t]
        pointsOnEdge = np.sort(out)
        pointsOnEdge = pointsOnEdge[1::2]  # Skip every other element to remove repeated elements
        for i in pointsOnEdge:
            maxMat[i]=(np.max(PDF[i]*GMat[i]))    
    else:
        for i in range(len(PDF)):
            maxMat[i]=(np.max(PDF[i]*GMat[i]))    
    addingAround = [np.asarray(maxMat) >= tolerance]
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
    while stillRemoving:
        boolZerosArray = checkIntegrandForZeroPoints(GMat, Pdf, removeZerosValuesIfLessThanTolerance,Mesh,tri, True)
    # if max(boolZerosArray == 1):
        ChangedBool=1
        for val in range(len(boolZerosArray)-1,-1,-1):
            if boolZerosArray[val] == 1: # remove the point
                GMat.pop(val)
                Mesh = np.delete(Mesh, val, 0)
                Grids.pop(val)
                Pdf = np.delete(Pdf, val, 0)
        else:
            stillRemoving = False
    Vertices, VerticesNum, tri = houseKeepingAfterRemovingPoints(Mesh, Grids, tri)
    for i in range(len(Mesh)-1,-1,-1):
        nearestPoint, index = UM.findNearestPoint(Mesh[i,0],Mesh[i,1], Mesh, True)            # print("Making Less Dense!...")
        distToNearestPoint = np.sqrt((nearestPoint[0,0] - Mesh[i,0])**2 + (nearestPoint[0,1] - Mesh[i,1])**2) 
        if distToNearestPoint > minDistanceBetweenPointsBoundary*2:
            GMat.pop(i)
            Mesh = np.delete(Mesh, i, 0)
            Grids.pop(i)
            Pdf = np.delete(Pdf, i, 0)
            ChangedBool = 1
    if ChangedBool == 1:
        Vertices, VerticesNum, tri = houseKeepingAfterRemovingPoints(Mesh, Grids, tri)
    print("Boundary points removed", length -len(Mesh))  
    return GMat, Mesh, Grids, Pdf, ChangedBool

def removeInteriorPointsToMakeLessDense(GMat, Mesh, Grids, Pdf, tri, boundaryOnlyBool):
    length = len(Mesh)
    Slopes = checkAddInteriorPoints(Mesh, Pdf)
    removePointsIfSlopeLessThanTolerance = 0.1 #np.quantile(Slopes,.1)
    pointsToRemove = np.asarray([np.asarray(Slopes) < removePointsIfSlopeLessThanTolerance]).T
    meshWithSmallSlopes = []
    ChangedBool=0
    corrIndices = [] # Indices in the bigger mesh for removal
    for i in range(len(Mesh)):
        if pointsToRemove[i]==1:
            corrIndices.append(i) #Index from bigger mesh
            meshWithSmallSlopes.append(Mesh[i,:])
    # plt.figure()
    # plt.plot(np.asarray(meshWithSmallSlopes)[:,0], np.asarray(meshWithSmallSlopes)[:,1], '.')
    # plt.show()
    meshWithSmallSlopes = np.asarray(meshWithSmallSlopes)
    kstep = distanceMetrics.fillDistance(meshWithSmallSlopes)
    print(kstep, maxDistanceBetweenPoints*(skipCount-1)/skipCount)
    if kstep < maxDistanceBetweenPoints*(skipCount-1)/skipCount:
        indices = LPR.getMeshIndicesToRemoveFromMesh(meshWithSmallSlopes, skipCount)
        corrIndices = np.sort(corrIndices)
        for j in range(len(indices)-1,-1,-1):
            nearestPoint, distances = UM.findNearestKPoints(Mesh[corrIndices[j],0],Mesh[corrIndices[j],1], meshWithSmallSlopes, 3)            # print("Making Less Dense!...")
            distToNearestPoint = np.max(distances)
            if distToNearestPoint < maxDistanceBetweenPoints:
                assert corrIndices[j] <= len(Mesh)
                assert corrIndices[j] <= len(Pdf)
                ChangedBool = 1
                GMat.pop(corrIndices[j])
                Mesh = np.delete(Mesh, corrIndices[j], 0)
                Grids.pop(corrIndices[j])
                Pdf = np.delete(Pdf, corrIndices[j], 0)
            else:
                print("Skip removing top of hill")

    else:
        print("Skipping making less dense", kstep)
    numReduced = length-len(Mesh)
    print("\n Removed ", numReduced, "to decrease density.")
    return GMat, Mesh, Grids, Pdf, ChangedBool

# Removes the flagged values from the list of mesh values and in Gmat. 
# boolZerosArray is the list of zeros and ones denoting which grid points to remove.
# Gmat, Mesh, Grids, Vertices, and VerticesNum are all used in the 2DTQ-UnorderedMesh method 
# and the parts associated with the removed points need to be removed.
def removePointsFromMesh(GMat, Mesh, Grids, Pdf, tri, boundaryOnlyBool):
    GMat, Mesh, Grids, Pdf, ChangedBool2 = removeBoundaryPoints(GMat, Mesh, Grids, Pdf, tri, boundaryOnlyBool)
    
    GMat, Mesh, Grids, Pdf, ChangedBool1 = removeInteriorPointsToMakeLessDense(GMat, Mesh, Grids, Pdf, tri, boundaryOnlyBool)
    # plt.figure()
    # plt.plot(Mesh[:,0],Mesh[:,1], '*')
    # plt.show()
    if ChangedBool1:
        Vertices, VerticesNum, tri = houseKeepingAfterRemovingPoints(Mesh, Grids, tri)
    ChangedBool = max(ChangedBool1, ChangedBool2)
    return GMat, Mesh, Grids, Pdf, ChangedBool


def houseKeepingAfterRemovingPoints(Mesh, Grids, tri):
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

# newPoints is a n by 2 vector with the x coordinates of the new mesh
#  in the first column and the y coordinates in the second column. 
def addPointsToMesh(Mesh, GMat, Grids, Vertices, VerticesNum, Pdf, triangulation, kstep, h, xmin, xmax, ymin, ymax):
    boundaryPointsToAddAround = checkIntegrandForAddingPointsAroundBoundaryPoints(GMat, Pdf, addPointsToBoundaryIfBiggerThanTolerance, Mesh, triangulation, True)
    Slopes = checkAddInteriorPoints(Mesh, Pdf)
    denisfyAroundPointIfSlopeLargerThanTolerance = 0.1 # np.quantile(Slopes,0.5)
    interiorPointsToAddAround = np.asarray([np.asarray(Slopes)> denisfyAroundPointIfSlopeLargerThanTolerance]).T
    meshWithBigSlopes = []
    ChangedBool=0
    for i in range(len(Mesh)):
        if interiorPointsToAddAround[i]==1:
            meshWithBigSlopes.append(Mesh[i,:])
    meshWithBigSlopes = np.asarray(meshWithBigSlopes)
    kstep = distanceMetrics.fillDistance(meshWithBigSlopes)
#    fig = plt.figure()
#    ax = Axes3D(fig)
#    ax.scatter(Mesh[:,0], Mesh[:,1], Slopes, c='r', marker='.')
    ChangedBool = 0
    # plt.figure()
    # plt.plot(Mesh[:,0],Mesh[:,1], '.k')
    # plt.show()
    numBoundaryAdded = 0
    numInteriorAdded = 0
    if max(boundaryPointsToAddAround == 1) or max(interiorPointsToAddAround == 1): 
        print("adding boundary points...")
        for val in range(len(boundaryPointsToAddAround)-1,-1,-1):
            if boundaryPointsToAddAround[val] == 1: # if we should extend boundary
                ChangedBool = 1
#                newPoints = addPointsRadially(Mesh[val,0], Mesh[val,1], Mesh, 4, kstep)
                allPoints, newPoints = LP.getLejaPointsWithStartingPoints(Mesh[val,0], Mesh[val,1], 3, Mesh, 3, np.sqrt(h)*fun.g1(),np.sqrt(h)*fun.g2(),6, 100)
                newPoints = checkIfDistToClosestPointIsOk(newPoints, Mesh, minDistanceBetweenPointsBoundary)
                for point in range(len(newPoints)):
                    Mesh = np.append(Mesh, np.asarray([[newPoints[point,0]],[newPoints[point,1]]]).T, axis=0)
                    xmin = np.min(Mesh[:,0]); xmax = np.max(Mesh[:,0])
                    ymin = np.min(Mesh[:,1]); ymax = np.max(Mesh[:,1])
                    #grid = UM.makeOrderedGridAroundPoint([newPoints[point,0],newPoints[point,1]],kstep, max(xmax-xmin, ymax-ymin), xmin,xmax,ymin,ymax)
                    grid = UM.makeOrderedGridAroundPoint([newPoints[point,0],newPoints[point,1]],kstep, max(xmax-xmin, ymax-ymin),newPoints[point,0]-6*np.sqrt(h)*fun.g1() ,newPoints[point,0]+6*np.sqrt(h)*fun.g1(),newPoints[point,1]-6*np.sqrt(h)*fun.g2(),newPoints[point,1]+6*np.sqrt(h)*fun.g1())

                    Grids.append(np.copy(grid))
                    Vertices.append([])
                    VerticesNum.append([])
                    for currGridPoint in range(len(grid)):
                        vertices, indices = UM.getVerticesForPoint([grid[currGridPoint,0], grid[currGridPoint,1]], Mesh, triangulation) # Points that make up triangle
                        Vertices[-1].append(np.copy(vertices))
                        VerticesNum[-1].append(np.copy(indices))
                    #pointVertices, pointIndices = UM.getVerticesForPoint([newPoints[point,0],newPoints[point,1]], Mesh, triangulation) # Points that make up triangle    
                    #threePdfVals = [Pdf[pointIndices[0]], Pdf[pointIndices[1]], Pdf[pointIndices[2]]]
                    #interp = UM.baryInterp([newPoints[point,0]],[newPoints[point,1]], pointVertices, threePdfVals)
                    Pdf = np.append(Pdf, 0) # Assume the new point has PDF value 0
                    gRow = generateGRow([newPoints[point,0], newPoints[point,1]], grid, kstep, h)
                    GMat.append(np.copy(gRow))
                    numBoundaryAdded = numBoundaryAdded +1
                    triangulation.add_points(np.asarray([[newPoints[point,0]],[newPoints[point,1]]]).T, restart=False)
            elif (kstep > minDistanceBetweenPoints) and (interiorPointsToAddAround[val] == 1): # if we should extend boundary
#                newPoints = addPointsRadially(Mesh[val,0], Mesh[val,1], Mesh, 4, kstep/2) 
                allPoints, newPoints = LP.getLejaPointsWithStartingPoints(Mesh[val,0], Mesh[val,1], 4, Mesh, 4, np.sqrt(h)*fun.g1(),np.sqrt(h)*fun.g2(), 6,100)
                newPoints = checkIfDistToClosestPointIsOk(newPoints, Mesh, minDistanceBetweenPoints)
                for point in range(len(newPoints)):
                    ChangedBool = 1
                    grid = UM.makeOrderedGridAroundPoint([newPoints[point,0],newPoints[point,1]],kstep, max(xmax-xmin, ymax-ymin),newPoints[point,0]-4*np.sqrt(h)*fun.g1() ,newPoints[point,0]+4*np.sqrt(h)*fun.g1(),newPoints[point,1]-4*np.sqrt(h)*fun.g2(),newPoints[point,1]+4*np.sqrt(h)*fun.g1())
                    Grids.append(np.copy(grid))
                    Vertices.append([])
                    VerticesNum.append([])
                    for currGridPoint in range(len(grid)):
                        vertices, indices = UM.getVerticesForPoint([grid[currGridPoint,0], grid[currGridPoint,1]], Mesh, triangulation) # Points that make up triangle
                        Vertices[-1].append(np.copy(vertices))
                        VerticesNum[-1].append(np.copy(indices))
                    pointVertices, pointIndices = UM.getVerticesForPoint([newPoints[point,0],newPoints[point,1]], Mesh, triangulation) # Points that make up triangle    
                    try: 
                        threePdfVals = [Pdf[pointIndices[0]], Pdf[pointIndices[1]], Pdf[pointIndices[2]]]
                        interp = UM.baryInterp([newPoints[point,0]],[newPoints[point,1]], pointVertices, threePdfVals)
                        # print(interp)
                        Pdf = np.append(Pdf, interp, axis=0)                      
                    except:
                        # print("WARNING: A boundary point may have been treated as an interior point")
                        Pdf = np.append(Pdf, [0], axis=0)
                    Mesh = np.append(Mesh, np.asarray([[newPoints[point,0]],[newPoints[point,1]]]).T, axis=0)
                    gRow = generateGRow([newPoints[point,0], newPoints[point,1]], grid, kstep, h)
                    GMat.append(np.copy(gRow))
                    numInteriorAdded = numInteriorAdded +1
                    triangulation.add_points(np.asarray([[newPoints[point,0]],[newPoints[point,1]]]).T, restart=False)
        
    print("# boundary points Added = ", numBoundaryAdded)    
    print("# interior points Added = ", numInteriorAdded)  
        
    return Mesh, GMat, Grids, Vertices, VerticesNum, Pdf, triangulation, ChangedBool, xmin, xmax, ymin, ymax 


def checkAddInteriorPoints(mesh, PDF):
    Slopes = []
    for i in range(len(mesh)):
        Px = mesh[i,0]
        Py = mesh[i,1]
        pdf1 = PDF[i]
        nearestPoint, index = UM.findNearestPoint(Px,Py, mesh, True)
        pdf2 = PDF[index]
        slope = (pdf1-pdf2)/(np.sqrt((Px-nearestPoint[0,0])**2 + (Py-nearestPoint[0,1])**2))
        Slopes.append(np.abs(slope))
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
    points = []
    for i in range(len(newPoints)):
        newPointX = newPoints[i,0]
        newPointY = newPoints[i,1]
        nearestPoint = UM.findNearestPoint(newPointX, newPointY, Mesh)
        distToNearestPoint = np.sqrt((nearestPoint[0,0] - newPointX)**2 + (nearestPoint[0,1] - newPointY)**2)
        if distToNearestPoint > minDist and distToNearestPoint < maxDistanceBetweenPoints:
            # print("adding")
            points.append([newPointX, newPointY])
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

