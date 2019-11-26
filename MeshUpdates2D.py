# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 10:38:41 2019

@author: Ryleigh
"""
import numpy as np
import Functions as fun
import UnorderedMesh as UM
from scipy.spatial import Delaunay

def checkIntegrandForZeroPoints(GMat, PDF, tolerance):
    maxMat = []
    for i in range(len(PDF)):
        maxMat.append(np.max(PDF[i]*GMat[i]))    
    possibleZeros = [np.asarray(maxMat) <= tolerance]
    return np.asarray(possibleZeros).T

def checkIntegralForZeroPoints(GMatRow, PDF, tolerance):
    newPDF = []
    for i in range(len(PDF)):
        newPDF.append(np.matmul(GMatRow,PDF))
    return [np.asarray(newPDF) <= tolerance]


def generateGRow(point, allPoints, kstep, h):
    row = []
    for i in range(len(allPoints)):
        val = kstep**2*fun.G(point[0], point[1], allPoints[i,0], allPoints[i,1], h)
        row.append(val)
    return row

# Removes the flagged values from the list of mesh values and in Gmat. 
# boolZerosArray is the list of zeros and ones denoting which grid points to remove.
# Gmat, Mesh, Grids, Vertices, and VerticesNum are all used in the 2DTQ-UnorderedMesh method 
# and the parts associated with the removed points need to be removed.
def removePointsFromMesh(boolZerosArray,GMat, Mesh, Grids, Vertices, VerticesNum, Pdf):
    ChangedBool = 0
    if max(boolZerosArray == 1):
        for val in range(len(boolZerosArray)-1,-1,-1):
            if boolZerosArray[val] == 1: # remove the point
                GMat.pop(val)
                Mesh = np.delete(Mesh, val, 0)
                Grids.pop(val)
#                Vertices.pop(val)
#                VerticesNum.pop(val)
                Pdf = np.delete(Pdf, val, 0)
               
                      
                ChangedBool=1
    print('removed # points')    
    print(np.sum(boolZerosArray))
    return GMat, Mesh, Grids, Vertices, VerticesNum, Pdf, ChangedBool
            

# newPoints is a n by 2 vector with the x coordinates of the new mesh
#  in the first column and the y coordinates in the second column. 
def addPointsToMesh(newPoints, Mesh, GMat, Grids, Vertices, VerticesNum, Pdf, triangulation, kstep, h, xmin, xmax, ymin, ymax):
    for point in range(len(newPoints[:,0])):
        triangulation.add_points(np.asarray([[newPoints[point,0]],[newPoints[point,1]]]).T, restart=False)
        Mesh = np.append(Mesh, np.asarray([[newPoints[point,0]],[newPoints[point,1]]]).T, axis=0)
        grid = UM.makeOrderedGridAroundPoint([newPoints[point,0],newPoints[point,1]],kstep, max(xmax-xmin, ymax-ymin), xmin,xmax,ymin,ymax)
        Grids.append(np.copy(grid))
        Vertices.append([])
        VerticesNum.append([])
        for currGridPoint in range(len(grid)):
            vertices, indices = UM.getVerticesForPoint([grid[currGridPoint,0], grid[currGridPoint,1]], Mesh, triangulation) # Points that make up triangle
            Vertices[point].append(np.copy(vertices))
            VerticesNum[point].append(np.copy(indices))
        pointVertices, pointIndices = UM.getVerticesForPoint([newPoints[point,0],newPoints[point,1]], Mesh, triangulation) # Points that make up triangle    
        interp = UM.baryInterp([newPoints[point,0]],[newPoints[point,1]], pointVertices, Pdf)
        np.append(Pdf, interp, axis=0)
        gRow = generateGRow([newPoints[point,0], newPoints[point,1]], grid, kstep, h)
        GMat.append(np.copy(gRow))
    return Mesh, GMat, Grids, Vertices, VerticesNum, Pdf, triangulation
        
    
#Mesh, GMat, Grids, Vertices, VerticesNum, Pdf, triangulation = addPointsToMesh(np.asarray([[0.5,0]]), mesh, GMat, Grids, Vertices, VerticesNum, pdf, tri, kstep, h, xmin, xmax, ymin, ymax)