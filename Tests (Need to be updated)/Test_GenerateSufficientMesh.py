import sys
sys.path.append('C:/Users/Rylei/Documents/SimpleDTQ')

from LejaQuadrature import getLejaQuadratureRule, getMeshValsThatAreClose, newIntegrand, getNewPDFVal, QuadratureByInterpolation
import numpy as np
from scipy.stats import multivariate_normal
from math import isclose
from scipy.stats import norm
from pyapprox.variables import IndependentMultivariateRandomVariable
from pyapprox.variable_transformations import AffineRandomVariableTransformation
from pyapprox.multivariate_polynomials import PolynomialChaosExpansion, define_poly_options_from_variable_transformation
from pyapprox.indexing import compute_hyperbolic_indices, tensor_product_indices,compute_tensor_product_level_indices
import GenerateLejaPoints as LP
from GenerateLejaPoints import getLejaSetFromPoints, generateLejaMesh, getLejaPoints, mapPointsBack, mapPointsTo
import UnorderedMesh as UM
import numpy as np
import matplotlib.pyplot as plt
from Functions import g1, g2
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy.spatial import Delaunay



def Test_GetLejaQuadratureRule():
    train_samples1, weights1 = getLejaQuadratureRule(0.1, 0.1 ,1,1)
    assert isclose(np.sum(weights1),1)
    train_samples, weights = getLejaQuadratureRule(0.1, 0.1,0,0)
    assert isclose(np.sum(weights),1)
    
    Aa1 = np.matmul(weights1, np.ones(len(train_samples1))) # should be 1
    assert isclose(Aa1[0], 1)
    Aa = np.matmul(weights, np.ones(len(train_samples))) 
    assert isclose(Aa[0], 1)
    
    rv = multivariate_normal([1, -1], [[0.25, 0], [0, 0.25]])
    vals1 = np.asarray([rv.pdf(train_samples1)]).T 
    vals = np.asarray([rv.pdf(train_samples)]).T
    Ac1 = np.matmul(weights1, vals1) # should be 0.000279332
    assert isclose(Ac1[0,0], 0.000279332, abs_tol = 1**(-8))
    Ac = np.matmul(weights, vals) # Should be 0.0130763
    assert isclose(Ac[0,0], 0.0130763, abs_tol = 1**(-8))
    print("GetLejaQuadratureRule - PASS")
    
    var = .01
    sigma=np.sqrt(var)
    train_samples, weights = getLejaQuadratureRule(sigma, sigma ,0,0)
    plt.figure()

    plt.scatter(train_samples[:,0], train_samples[:,1], c='r')
    
    assert isclose(np.sum(weights),1, abs_tol=1**(-5))
    rv = multivariate_normal([0, 0], [[var, 0], [0, var]])
    vals = np.asarray([rv.pdf(train_samples)]).T
    # vals =np.ones(len(vals))

    Ac11 = np.matmul(weights, vals) 
    print(Ac11)

import distanceMetrics as DM  
def Test_LejaQuadratureOnLejaPoints():
    mesh = LP.generateLejaMesh(230, .1, .1, 30)
    h = 0.01
    var = .01
    sigma=np.sqrt(var)
    
    ii = 0
    muX = mesh[ii,0]
    muY = mesh[ii,1]
  
    print(muX)
    print(muY)
    # print(DM.fillDistance(mesh))
    # plt.figure()
    # plt.scatter(mesh[:,0], mesh[:,1], c='r')
    
    pdf = UM.generateICPDF(mesh[:,0], mesh[:,1], sigma, sigma)
    value = QuadratureByInterpolation(mesh, pdf, sigma, sigma, muX, muY, 20)
    print(value)

    
    # pdf = mesh[:,0]**2+mesh[:,1]**2
    # value = QuadratureByInterpolation(mesh, pdf, sigma, sigma, muX, muY, 4)
    
from mpl_toolkits.mplot3d import Axes3D

def Test_LejaQuadratureLinearizationOnLejaPointsIfAllPointsKnownOrig():
    h = 0.01
    sigmaX=np.sqrt(h)*g1()
    sigmaY=np.sqrt(h)*g2()
    sigma = np.sqrt(h)

    mesh = LP.generateLejaMesh(230, sigma, sigma, 20)
    
    newPDF = []
    condNums = []
    for ii in range(len(mesh)):
    # for ii in range(len(mesh)):
        muX = mesh[ii,0]
        muY = mesh[ii,1]
        mesh1 = mapPointsBack(muX, muY, mesh, 1, 1)
        
        rv = multivariate_normal([0, 0], [[sigma**2, 0], [0, sigma**2]])
        pdf = np.asarray([rv.pdf(mesh1)]).T
        integrand = newIntegrand(muX, muY, mesh1, h)
        testing = np.squeeze(pdf)*integrand
        
        
        value, condNum= QuadratureByInterpolation(mesh1, testing, sigmaX, sigmaY, muX, muY, 20)
            
        print(value)
        newPDF.append(value)
        condNums.append(condNum)
    newPDF = np.asarray(newPDF)
    condNums = np.asarray([condNums]).T
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(mesh[:,0], mesh[:,1], newPDF, c='r', marker='.')
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(mesh1[:,0], mesh1[:,1], testing, c='k', marker='.')
    return newPDF,condNums, mesh


import distanceMetrics as dm
from scipy.spatial import Delaunay
import MeshUpdates2D as meshup
from scipy.interpolate import griddata

def checkDist(mesh, newMesh, minDist):
    points = []
    for i in range(len(newMesh)):
        newPointX = newMesh[i,0]
        newPointY = newMesh[i,1]
        nearestPoint = UM.findNearestPoint(newPointX, newPointY, mesh)
        distToNearestPoint = np.sqrt((nearestPoint[0,0] - newPointX)**2 + (nearestPoint[0,1] - newPointY)**2)
        if distToNearestPoint > minDist:
            points.append([newPointX, newPointY])
            
    return np.asarray(points)


def getSufficientMesh(muX, muY, sigmaX, sigmaY, currMesh, currPDF):
    dist = dm.fillDistanceAvg(currMesh) 
    
    currMesh2 = np.copy(currMesh)
    tri = Delaunay(currMesh)
    
    pointsOnEdge = meshup.getBoundaryPoints(currMesh, tri, 0.1)
    newMeshPoints = []
    newPdfVals = []
    pp = LP.generateLejaMesh(100, sigmaX, sigmaY, 20)

    for i in range(len(pointsOnEdge)):
        print(i, len(pointsOnEdge))
        suffMesh = mapPointsBack(currMesh[pointsOnEdge[i],0], currMesh[pointsOnEdge[i],1], pp, 1, 1)
        # plt.figure()
        # plt.scatter(suffMesh[:,0], suffMesh[:,1], c= 'r')
        # plt.scatter(currMesh[:,0], currMesh[:,1], c='k')
        newpp = checkDist(currMesh, suffMesh, dist)

        NewVals = np.asarray(newpp)
     
        if len(NewVals >0):
            currMesh2 = np.vstack((currMesh2, NewVals))
            if len(newMeshPoints) == 0:
                newMeshPoints = NewVals
            else:
                newMeshPoints = np.vstack((newMeshPoints, NewVals))
          
            

    interp = griddata(currMesh, currPDF, newMeshPoints, method='cubic', fill_value=10**(-10))
    fullPDF = np.vstack((currPDF, interp))
    plt.figure()
    plt.scatter(currMesh2[:,0], currMesh2[:,1], c= 'r')
    plt.scatter(currMesh[:,0], currMesh[:,1], c='k')
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(currMesh2[:,0], currMesh2[:,1], fullPDF, c='r', marker='.')
    
    return NewVals, interp

def getSufficientMeshFast(muX, muY, sigmaX, sigmaY, currMesh, currPDF):
    dist = dm.fillDistanceAvg(currMesh) 
    meshBigger = LP.generateLejaMesh(200, sigmaX*2, sigmaY*2, 50)
    meshBigger1 = LP.generateLejaMesh(200, sigmaX*2.5, sigmaY*2.5, 50)
    meshBigger2 = LP.generateLejaMesh(200, sigmaX*1.5, sigmaY*1.5, 50)
    meshNew = np.vstack((meshBigger, meshBigger1))
    meshNew = np.vstack((meshNew, meshBigger2))


    keep = checkDist(currMesh, meshNew, 0.1)
    plt.figure()
    plt.scatter(currMesh[:,0], currMesh[:,1] , c='g')
    plt.scatter(keep[:,0], keep[:,1] , c='r')
    plt.show()
    
    
    return keep
        


from scipy import interpolate

h = 0.01
sigmaX=np.sqrt(h)*g1()
sigmaY=np.sqrt(h)*g2()
sigma = np.sqrt(h)

mesh = LP.generateLejaMesh(230, sigmaX, sigmaY, 20)
rv = multivariate_normal([0, 0], [[sigma**2, 0], [0, sigma**2]])
pdf = np.asarray([rv.pdf(mesh)]).T
# plt.figure()
# plt.scatter(mesh[:,0], mesh[:,1] , c='g')
# plt.show()

meshNew = getSufficientMeshFast(0, 0, sigmaX, sigmaY, mesh, pdf)
meshNew = np.vstack((mesh,meshNew))
pdfNewVals = np.asarray([rv.pdf(meshNew)]).T
# pdfNew = np.vstack((pdf, np.zeros((len(meshNew),1))))
pdfNew = np.vstack((pdf, pdfNewVals))

plt.figure()
plt.scatter(meshNew[:,0], meshNew[:,1] , c='g')
plt.show()

meshNew = mesh
pdfNew = pdf

newPDF = []
condNums = []
# rv = multivariate_normal([0, 0], [[sigma**2, 0], [0, sigma**2]])
# pdf = np.asarray([rv.pdf(mesh)]).T
for ii in range(len(mesh)):
    print('########################',ii/len(mesh)*100, '%')
    muX = mesh[ii,0]
    muY = mesh[ii,1]

    meshTemp = np.delete(meshNew, ii, axis=0)
    pdfTemp = np.delete(pdfNew, ii, axis=0)
    
    mesh1, indices = getLejaSetFromPoints(muX, muY, meshTemp, 130, 20, sigmaX, sigmaY)
    meshTemp = np.vstack(([muX,muY], meshTemp))
    pdfTemp = np.vstack((pdf[ii], pdfTemp))
    pdfs = []
    Pxs = []
    Pys = []
    for i in range(len(indices)):
        pdfs.append(pdfTemp[indices[i]])
        Pxs.append(meshTemp[indices[i],0])
        Pys.append(meshTemp[indices[i],1])
    pdfNew1 = np.asarray(pdfs)
    mesh1 = np.vstack((Pxs, Pys))
    mesh1 = np.asarray(mesh1).T
    
    
    # print(len(mesh1))
    integrand = newIntegrand(muX, muY, mesh1, h)
    testing = np.squeeze(pdfNew1)*integrand
    
    value, condNum= QuadratureByInterpolation(mesh1, testing, sigmaX, sigmaY, muX, muY, 20)
    print(value)
    print(condNum)
    
    newPDF.append(value)
    condNums.append(condNum)
# print(countUseMorePoints)
newPDF = np.asarray(newPDF)
condNums = np.asarray([condNums]).T
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(mesh[:,0], mesh[:,1], newPDF, c='r', marker='.')

# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(mesh1[:,0], mesh1[:,1], testing, c='k', marker='.')

plt.figure()
# plt.semilogy(np.sqrt(mesh[:,0]**2 +mesh[:,1]**2), condNums, '.')
plt.semilogy(np.sqrt(Meshes[-1][:,0]**2 + Meshes[-1][:,1]**2), condnums, '.')

plt.xlabel("Distance from center of mesh")
plt.ylabel("log of condition number")