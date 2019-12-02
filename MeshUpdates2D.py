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
    

def alpha_shape(points, alpha, only_outer=True):
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

    tri = Delaunay(points)
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
    return edges

from matplotlib.pyplot import *
import numpy as np
# Constructing the input point data
np.random.seed(0)
x = 3.0 * np.random.rand(2000)
y = 2.0 * np.random.rand(2000) - 1.0
inside = ((x ** 2 + y ** 2 > 1.0) & ((x - 3) ** 2 + y ** 2 > 1.0))

points = np.vstack([x[inside], y[inside]]).T

# Computing the alpha shape
edges = alpha_shape(points, alpha=0.25, only_outer=True)

# Plotting the output
figure()
axis('equal')
plot(points[:, 0], points[:, 1], '.')
for i, j in edges:
    plot(points[[i, j], 0], points[[i, j], 1])
show()