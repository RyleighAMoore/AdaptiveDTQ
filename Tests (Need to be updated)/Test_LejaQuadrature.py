

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
from GenerateLejaPoints import getLejaSetFromPoints, getLejaPoints, mapPointsBack, mapPointsTo
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

# def checkDist(mesh, newMesh, minDist):
#     points = []
#     for i in range(len(newMesh)):
#         newPointX = newMesh[i,0]
#         newPointY = newMesh[i,1]
#         nearestPoint = UM.findNearestPoint(newPointX, newPointY, mesh)
#         distToNearestPoint = np.sqrt((nearestPoint[0,0] - newPointX)**2 + (nearestPoint[0,1] - newPointY)**2)
#         if distToNearestPoint > minDist:
#             points.append([newPointX, newPointY])
            
#     return np.asarray(points)


# def getSufficientMesh(muX, muY, sigmaX, sigmaY, currMesh, currPDF):
#     dist = dm.fillDistanceAvg(currMesh) 
    
#     currMesh2 = np.copy(currMesh)
#     tri = Delaunay(currMesh)
    
#     pointsOnEdge = meshup.getBoundaryPoints(currMesh, tri, 0.1)
#     newMeshPoints = []
#     newPdfVals = []
#     pp = LP.generateLejaMesh(100, sigmaX, sigmaY, 20)

#     for i in range(len(pointsOnEdge)):
#         print(i, len(pointsOnEdge))
#         suffMesh = mapPointsBack(currMesh[pointsOnEdge[i],0], currMesh[pointsOnEdge[i],1], pp, 1, 1)
#         # plt.figure()
#         # plt.scatter(suffMesh[:,0], suffMesh[:,1], c= 'r')
#         # plt.scatter(currMesh[:,0], currMesh[:,1], c='k')
#         newpp = checkDist(currMesh, suffMesh, dist)

#         NewVals = np.asarray(newpp)
     
#         if len(NewVals >0):
#             currMesh2 = np.vstack((currMesh2, NewVals))
#             if len(newMeshPoints) == 0:
#                 newMeshPoints = NewVals
#             else:
#                 newMeshPoints = np.vstack((newMeshPoints, NewVals))
          
            

#     interp = griddata(currMesh, currPDF, newMeshPoints, method='cubic', fill_value=10**(-10))
#     fullPDF = np.vstack((currPDF, interp))
#     plt.figure()
#     plt.scatter(currMesh2[:,0], currMesh2[:,1], c= 'r')
#     plt.scatter(currMesh[:,0], currMesh[:,1], c='k')
    
#     fig = plt.figure()
#     ax = Axes3D(fig)
#     ax.scatter(currMesh2[:,0], currMesh2[:,1], fullPDF, c='r', marker='.')
    
#     return NewVals, interp




# def Test_LejaQuadratureLinearizationOnLejaPointsSetStartingMesh():
#     h = 0.01
#     sigmaX=np.sqrt(h)*g1()
#     sigmaY=np.sqrt(h)*g2()
#     sigma = np.sqrt(h)

#     meshOld = LP.generateLejaMesh(230, sigmaX, sigmaY, 20)
#     rv = multivariate_normal([0, 0], [[sigma**2, 0], [0, sigma**2]])
#     pdfOld = np.asarray([rv.pdf(meshOld)]).T
#     mesh, pdf = getSufficientMesh(0, 0, sigmaX, sigmaY, meshOld, pdfOld)

#     # mesh, new = getLejaPoints(230, np.asarray([[0],[0]]), 20, num_candidate_samples = 5000, dimensions=2, defCandidateSamples=False, candidateSampleMesh = [], returnIndices = False)
#     # plt.figure()
#     # plt.scatter(mesh[:,0], mesh[:,1], c='k', marker='*')

    
#     newPDF = []
#     condNums = []
    
#     for ii in range(len(meshOld)):
#         print('########################',ii/len(mesh)*100, '%')
#         muX = mesh[ii,0]
#         muY = mesh[ii,1]
        
        
#         # mesh12, pdfNew1 = getMeshValsThatAreClose(mesh, pdf, sigmaX, sigmaY, muX, muY)

#         # mesh1, pdfNew1 = getMeshValsThatAreClose(mesh, pdf, sigmaX, sigmaY, muX, muY)
#         meshTemp = np.delete(mesh, ii, axis=0)
#         pdfTemp = np.delete(pdf, ii, axis=0)
        
#         mesh1, indices = getLejaSetFromPoints(muX, muY, meshTemp, 130, 20, sigmaX, sigmaY)
#         meshTemp = np.vstack(([muX,muY], meshTemp))
#         pdfTemp = np.vstack((pdf[ii], pdfTemp))
#         pdfNew = []
#         Pxs = []
#         Pys = []
#         for i in range(len(indices)):
#             pdfNew.append(pdfTemp[indices[i]])
#             Pxs.append(meshTemp[indices[i],0])
#             Pys.append(meshTemp[indices[i],1])
#         pdfNew1 = np.asarray(pdfNew)
#         mesh1 = np.vstack((Pxs, Pys))
#         mesh1 = np.asarray(mesh1).T
        
        
#         print(len(mesh1))
#         integrand = newIntegrand(muX, muY, mesh1, h)
#         testing = np.squeeze(pdfNew1)*integrand
        
#         value, condNum= QuadratureByInterpolation(mesh1, testing, sigmaX, sigmaY, muX, muY, 20)
#         print(value)
#         if condNum > 2 or value < 0:
#             mesh12, pdfNew1 = getMeshValsThatAreClose(mesh, pdf, sigmaX, sigmaY, muX, muY)
#             # mesh12 = np.vstack(([muX,muY], mesh12))
#             needPoints = 130
#             if len(mesh12) < needPoints:
#                 mesh12 = mapPointsTo(muX, muY, mesh12, 1/sigmaX, 1/sigmaY)
#                 num_leja_samples = 200
#                 initial_samples = mesh12
#                 numBasis=20
#                 allp, new = LP.getLejaPoints(num_leja_samples, initial_samples.T,numBasis, num_candidate_samples = 230, dimensions=2, defCandidateSamples=False, candidateSampleMesh = [], returnIndices = False)
#                 mesh12 = mapPointsBack(muX,muY, allp, sigmaX, sigmaY)
            
            
#             # plt.figure()
#             # plt.scatter(mesh[:,0], mesh[:,1], c='k', marker='*')
#             # plt.scatter(mesh12[:,0], mesh12[:,1], c='r', marker='.')
#             # plt.scatter(muX, muY, c='g', marker='.')
    
#             pdfNew = np.asarray(griddata(mesh, pdf, mesh12, method='cubic', fill_value=0))
#             pdfNew[pdfNew < 0] = 0
#             integrand = newIntegrand(muX, muY, mesh12, h)
#             testing = np.squeeze(pdfNew)*integrand
            
            
#             value, condNum= QuadratureByInterpolation(mesh12, testing, sigmaX, sigmaY, muX, muY, 20)
#             print(value)
#             if value<0:
#                 value= [10**(-10)]
#                 # print('#######################################', muX, muY)
#                 # plt.figure()
#                 # plt.scatter(mesh[:,0], mesh[:,1], c='k', marker='*')
#                 # plt.scatter(mesh12[:,0], mesh12[:,1], c='r', marker='.')
#                 # plt.scatter(muX, muY, c='g', marker='.')
        
#         newPDF.append(value)
#         condNums.append(condNum)
#     newPDF = np.asarray(newPDF)
#     condNums = np.asarray([condNums]).T
#     fig = plt.figure()
#     ax = Axes3D(fig)
#     ax.scatter(mesh[:,0], mesh[:,1], newPDF, c='r', marker='.')
    
#     # fig = plt.figure()
#     # ax = Axes3D(fig)
#     # ax.scatter(mesh1[:,0], mesh1[:,1], testing, c='k', marker='.')
#     return newPDF,condNums, mesh




import GenerateLejaPoints as LP


def Test_LejaQuadratureLinearizationOnLejaPoints():
    h = 0.01
    sigmaX=np.sqrt(h)*g1()
    sigmaY=np.sqrt(h)*g2()
    sigma = np.sqrt(h)

    mesh = LP.generateLejaMesh(230, sigmaX, sigmaY, 20)
    
    newPDF = []
    condNums = []
    rv = multivariate_normal([0, 0], [[sigma**2, 0], [0, sigma**2]])
    pdf = np.asarray([rv.pdf(mesh)]).T
    countUseMorePoints = 0
    for ii in range(len(mesh)):
        # print('########################',ii/len(mesh)*100, '%')
        muX = mesh[ii,0]
        muY = mesh[ii,1]

        # mesh1, pdfNew1 = getMeshValsThatAreClose(mesh, pdf, sigmaX, sigmaY, muX, muY)
        meshTemp = np.delete(mesh, ii, axis=0)
        pdfTemp = np.delete(pdf, ii, axis=0)
        
        mesh1, indices = getLejaSetFromPoints(muX, muY, meshTemp, 130, 20, sigmaX, sigmaY)
        meshTemp = np.vstack(([muX,muY], meshTemp))
        pdfTemp = np.vstack((pdf[ii], pdfTemp))
        pdfNew = []
        Pxs = []
        Pys = []
        for i in range(len(indices)):
            pdfNew.append(pdfTemp[indices[i]])
            Pxs.append(meshTemp[indices[i],0])
            Pys.append(meshTemp[indices[i],1])
        pdfNew1 = np.asarray(pdfNew)
        mesh1 = np.vstack((Pxs, Pys))
        mesh1 = np.asarray(mesh1).T
        
        
        print(len(mesh1))
        integrand = newIntegrand(muX, muY, mesh1, h)
        testing = np.squeeze(pdfNew1)*integrand
        
        value, condNum= QuadratureByInterpolation(mesh1, testing, sigmaX, sigmaY, muX, muY, 20)
        print(value)
        if condNum > 5 or value < 0:
            countUseMorePoints = countUseMorePoints+1
            mesh12, pdfNew1 = getMeshValsThatAreClose(mesh, pdf, sigmaX, sigmaY, muX, muY)
            # mesh12 = np.vstack(([muX,muY], mesh12))
            needPoints = 130
            if len(mesh12) < needPoints:
                mesh12 = mapPointsTo(muX, muY, mesh12, 1/sigmaX, 1/sigmaY)
                num_leja_samples = 130
                initial_samples = mesh12
                numBasis=20
                allp, new = LP.getLejaPoints(num_leja_samples, initial_samples.T,numBasis, num_candidate_samples = 230, dimensions=2, defCandidateSamples=False, candidateSampleMesh = [], returnIndices = False)
                mesh12 = mapPointsBack(muX,muY, allp, sigmaX, sigmaY)
            
            
            # plt.figure()
            # plt.scatter(mesh[:,0], mesh[:,1], c='k', marker='*')
            # plt.scatter(mesh12[:,0], mesh12[:,1], c='r', marker='.')
            # plt.scatter(muX, muY, c='g', marker='.')
    
            pdfNew = np.asarray(griddata(mesh, pdf, mesh12, method='cubic', fill_value=0))
            pdfNew[pdfNew < 0] = 0
            integrand = newIntegrand(muX, muY, mesh12, h)
            testing = np.squeeze(pdfNew)*integrand
            
            
            value, condNum= QuadratureByInterpolation(mesh12, testing, sigmaX, sigmaY, muX, muY, 20)
            print(value)
            if value<0:
                value= [10**(-10)]
                # print('#######################################', muX, muY)
                # plt.figure()
                # plt.scatter(mesh[:,0], mesh[:,1], c='k', marker='*')
                # plt.scatter(mesh12[:,0], mesh12[:,1], c='r', marker='.')
                # plt.scatter(muX, muY, c='g', marker='.')
        
        newPDF.append(value)
        condNums.append(condNum)
    print(countUseMorePoints)
    newPDF = np.asarray(newPDF)
    condNums = np.asarray([condNums]).T
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(mesh[:,0], mesh[:,1], newPDF, c='r', marker='.')
    
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(mesh1[:,0], mesh1[:,1], testing, c='k', marker='.')
    return newPDF,condNums, mesh



    

# Test_GetLejaQuadratureRule()
# Test_LejaQuadratureOnLejaPoints()
# newPDF,condNums, meshVals = Test_LejaQuadratureLinearizationOnLejaPoints()
