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


import GenerateLejaPoints as LP
def Test_LejaQuadratureLinearizationOnLejaPointsIfAllPointsKnown():
    h = 0.01
    sigmaX=np.sqrt(h)*g1()
    sigmaY=np.sqrt(h)*g2()
    sigma = np.sqrt(h)

    mesh = LP.generateLejaMesh(230, sigma, sigma, 20)
    # mesh, new = getLejaPoints(230, np.asarray([[0],[0]]), 20, num_candidate_samples = 5000, dimensions=2, defCandidateSamples=False, candidateSampleMesh = [], returnIndices = False)
    # plt.figure()
    # plt.scatter(mesh[:,0], mesh[:,1], c='k', marker='*')

    
    newPDF = []
    condNums = []
    rv = multivariate_normal([0, 0], [[sigma**2, 0], [0, sigma**2]])
    pdf = np.asarray([rv.pdf(mesh)]).T
    for ii in range(len(mesh)):
        print('########################',ii/len(mesh)*100, '%')
    # for ii in range(len(mesh)):
        # muX = np.min(mesh[:,0])
        # muY = np.min(mesh[:,1])
        muX = mesh[ii,0]
        muY = mesh[ii,1]
        # mesh1 = mapPointsBack(muX, muY, mesh, 1, 1)
        
        integrand = newIntegrand(muX, muY, mesh, h)
        testing = np.squeeze(pdf)*integrand
        value, condNum= QuadratureByInterpolation(mesh, testing, sigmaX, sigmaY, muX, muY, 20)
        if condNum > 1.5 or value<0:
            mesh12, pdfNew1 = getMeshValsThatAreClose(mesh, pdf, sigmaX, sigmaY, muX, muY)
            # mesh12 = np.vstack(([muX,muY], mesh12))
            mesh12 = mapPointsTo(muX, muY, mesh12, 1/sigmaX, 1/sigmaY)
            needPoints = 230
            if len(mesh12) < needPoints:
                num_leja_samples = len(mesh)
                initial_samples = mesh12
                numBasis=40
                allp, new = LP.getLejaPoints(num_leja_samples, initial_samples.T,numBasis, num_candidate_samples = 500, dimensions=2, defCandidateSamples=False, candidateSampleMesh = [], returnIndices = False)
                mesh12 = mapPointsBack(muX,muY, allp, sigmaX, sigmaY)
            else:
                print("normal")
            
            # plt.figure()
            # plt.scatter(mesh[:,0], mesh[:,1], c='k', marker='*')
            # plt.scatter(mesh12[:,0], mesh12[:,1], c='r', marker='.')
            # plt.scatter(muX, muY, c='g', marker='.')
    
            pdfNew = np.asarray(griddata(mesh, pdf, mesh12, method='cubic', fill_value=0))
            pdfNew[pdfNew < 0] = 0
            integrand = newIntegrand(muX, muY, mesh12, h)
            testing = np.squeeze(pdfNew)*integrand
            
            
            value, condNum= QuadratureByInterpolation(mesh12, testing, sigmaX, sigmaY, muX, muY, 20)
            if value<0:
                value= [10**(-20)]
                # print('#######################################', muX, muY)
                # plt.figure()
                # plt.scatter(mesh[:,0], mesh[:,1], c='k', marker='*')
                # plt.scatter(mesh12[:,0], mesh12[:,1], c='r', marker='.')
                # plt.scatter(muX, muY, c='g', marker='.')
        print(value)
        newPDF.append(value)
        condNums.append(condNum)
    newPDF = np.asarray(newPDF)
    condNums = np.asarray([condNums]).T
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(mesh[:,0], mesh[:,1], newPDF, c='r', marker='.')
    
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(mesh1[:,0], mesh1[:,1], testing, c='k', marker='.')
    return newPDF,condNums, mesh


from scipy.spatial import Delaunay

import MeshUpdates2D as meshup
def getSufficientMesh2(muX, muY, sigmaX, sigmaY, currMesh):
    tri = Delaunay(currMesh)
    # pointsOnEdge = meshup.getBoundaryPoints(currMesh, tri, 0.01)

    suffMesh = LP.generateLejaMesh(230, sigmaX*1.5, sigmaY*1.5, 20)
    suffMesh = mapPointsBack(muX, muY, suffMesh, 1, 1)
    

    p = np.array(suffMesh)
    vals = tri.find_simplex(p)
    NewVals = []
    for i in range(len(vals)):
        if vals[i] == -1:
            NewVals.append(suffMesh[i,:])
    NewVals = np.asarray(NewVals)
    plt.figure()
    plt.scatter(currMesh[:,0], currMesh[:,1])
    plt.scatter(NewVals[:,0], NewVals[:,1])
    
    return NewVals



def checkDist(newPoints, Mesh, minDist):
    '''Checks to make sure that a new point we want to add is not too close or too far from another points'''
    points = []
    for i in range(len(newPoints)):
        newPointX = newPoints[i,0]
        newPointY = newPoints[i,1]
        nearestPoint = UM.findNearestPoint(newPointX, newPointY, Mesh)
        distToNearestPoint = np.sqrt((nearestPoint[0,0] - newPointX)**2 + (nearestPoint[0,1] - newPointY)**2)
        if distToNearestPoint > minDist:
            points.append([newPointX, newPointY])
        # else:
        #     print("Nope")
    return np.asarray(points)


import numpy as np
from scipy.optimize import linprog

def in_hull(points, x):
    n_points = len(points)
    n_dim = len(x)
    c = np.zeros(n_points)
    A = np.r_[points.T,np.ones((1,n_points))]
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success

# n_points = 10000
# n_dim = 2
# Z = np.random.rand(n_points,n_dim)
# x = np.random.rand(n_dim)
# print(in_hull(Z, x))
import distanceMetrics as dm
def getSufficientMesh(muX, muY, sigmaX, sigmaY, currMesh):
    dist = dm.fillDistanceAvg(currMesh) 
    print(dist)
    currMesh2 = np.copy(currMesh)
    tri = Delaunay(currMesh)
    
    pointsOnEdge = meshup.getBoundaryPoints(currMesh, tri, 0.1)

    addPoints = []
    for i in range(len(pointsOnEdge)):
        pp = LP.generateLejaMesh(50, sigmaX, sigmaY, 20)
        suffMesh = mapPointsBack(currMesh[pointsOnEdge[i],0], currMesh[pointsOnEdge[i],1], pp, 1, 1)
        newpp = checkDist(suffMesh, currMesh2, dist)
        p = np.asarray(suffMesh)
        NewVals = np.asarray(newpp)
        # plt.figure()
        # plt.scatter(currMesh2[:,0], currMesh2[:,1])
        # plt.scatter(p[:,0], p[:,1])
        # plt.scatter(currMesh[pointsOnEdge[i],0], currMesh[pointsOnEdge[i],1], c='r')
        # NewVals = []
        # for c in range(len(newpp)):
        #     gg = in_hull(currMesh, newpp[c])
        #     if not gg:
        #         NewVals.append(np.copy(newpp[c,:]))
            
        # vals = tri.find_simplex(p)
        # NewVals = []
        # if len(vals >0) and np.min(vals)==-1:
        #     for i in range(len(vals)):
        #         if vals[i] == -1:
        #             NewVals.append(suffMesh[i,:])
        if len(NewVals >0):
            NewVals = np.asarray(NewVals)
            currMesh2 = np.vstack((currMesh2, NewVals))
    
    # suffMesh = LP.generateLejaMesh(500, sigmaX*1.5, sigmaY*1.5, 50)
    # suffMesh = mapPointsBack(muX, muY, suffMesh, 1, 1)

    
    plt.figure()
    plt.scatter(currMesh2[:,0], currMesh2[:,1], c= 'r')
    plt.scatter(currMesh[:,0], currMesh[:,1], c='k')
    
    return currMesh2

from scipy.interpolate import griddata
def getSufficientMesh3(muX, muY, sigmaX, sigmaY, currMesh, currPDF):
    dist = dm.fillDistanceAvg(currMesh) 
    # print(dist)
    currMesh2 = np.copy(currMesh)
    addPoints = []
    pp = LP.generateLejaMesh(230, sigmaX, sigmaY, 20)
    suffMesh = mapPointsBack(muX, muY, pp, 1, 1)
    newpp = checkDist(suffMesh, currMesh2, dist)
    NewVals = np.asarray(newpp)
    plt.figure()
    plt.scatter(currMesh2[:,0], currMesh2[:,1])
    plt.scatter(np.asarray(newpp)[:,0], np.asarray(newpp)[:,1])
    plt.scatter(muX, muY, c='r')
    # NewVals = []
    # for c in range(len(newpp)):
    #     gg = in_hull(currMesh, newpp[c])
    #     if not gg:
    #         NewVals.append(np.copy(newpp[c,:]))
        
    # vals = tri.find_simplex(p)
    # NewVals = []
    # if len(vals >0) and np.min(vals)==-1:
    #     for i in range(len(vals)):
    #         if vals[i] == -1:
    #             NewVals.append(suffMesh[i,:])
    if len(NewVals >0):
        # interp = np.asarray(griddata(currMesh, currPDF, NewVals, method='cubic', fill_value=0))
        rv = multivariate_normal([0, 0], [[sigmaX**2, 0], [0, sigmaY**2]])
        interp = np.asarray([rv.pdf(NewVals)]).T
        currMesh2 = np.vstack((currMesh2, NewVals))
        
    
    # suffMesh = LP.generateLejaMesh(500, sigmaX*1.5, sigmaY*1.5, 50)
    # suffMesh = mapPointsBack(muX, muY, suffMesh, 1, 1)

    
    # plt.figure()
    # plt.scatter(currMesh2[:,0], currMesh2[:,1], c= 'r')
    # plt.scatter(currMesh[:,0], currMesh[:,1], c='k')

    
    return NewVals, interp
        

def Test_getSufficientMesh():
    h = 0.01
    muX =0
    muY= 0
    sigmaX=np.sqrt(h)*g1()
    sigmaY=np.sqrt(h)*g2()
    sigma = np.sqrt(h)
    currMesh = LP.generateLejaMesh(230, sigmaX, sigmaY, 20)
    newVals = getSufficientMesh(muX, muY, sigmaX, sigmaY, currMesh)
    return newVals

# vals = Test_getSufficientMesh()

from tqdm import tqdm, trange
import math
def Test_LejaQuadratureLinearizationOnLejaPoints():
    h = 0.01
    sigmaX=np.sqrt(h)*g1()
    sigmaY=np.sqrt(h)*g2()
    sigma = np.sqrt(h)

    mesh = LP.generateLejaMesh(230, sigmaX, sigmaY, 30)
    # newMeshPoints = getSufficientMesh(0, 0, sigmaX, sigmaY, mesh)
    
    # # newMeshPoints = np.vstack((mesh, newMeshPoints))
    # newVals = np.zeros(len(newMeshPoints)-len(mesh))

    rv = multivariate_normal([0, 0], [[sigma**2, 0], [0, sigma**2]])
    pdf = np.asarray([rv.pdf(mesh)]).T
    
    # newVals = np.hstack((np.squeeze(pdf), newVals))


    # mesh = newMeshPoints
    # pdf = np.asarray([newVals]).T
    newPDF = []
    condNums = []
    for ii in range(len(mesh)):
        print('########################',ii/len(mesh)*100, '%')
    # for ii in range(len(mesh)):
        muX = mesh[ii,0]
        muY = mesh[ii,1]
        # print(muX)
        # print(muY)
        # mesh2 = mapPointsTo(muX, muY, mesh, 1/sigmaX, 1/sigmaY)
        # meshclose, pdfclose = getMeshValsThatAreClose(mesh, pdf, sigmaX, sigmaY, muX, muY)
        meshclose = np.copy(mesh)
        pdfclose = np.copy(pdf)
        # plt.figure()
        # plt.scatter(meshclose[:,0], meshclose[:,1])
        # plt.scatter(mesh[:,0], mesh[:,1], c='r', marker='*')
        # plt.scatter(muX, muY, c='b')
        
        
        # candidate_samples = np.delete(meshclose, ii, axis=0)
        # cand_pdf = np.delete(pdfclose, ii, axis=0)
        # numPoints = min(230, math.floor(1*len(meshclose)))
        # mesh1, indices = getLejaSetFromPoints(muX, muY, candidate_samples, numPoints, 20)
        # # mesh1 = mapPointsBack(muX, muY, mesh1, sigmaX, sigmaY)
        # candidate_samples = np.vstack(([muX,muY], candidate_samples))
        # cand_pdf = np.vstack((pdfclose[ii], cand_pdf))
        
        # # plt.figure()
        # # plt.scatter(mesh[:,0], mesh[:,1])
        # # plt.scatter(mesh1[:,0], mesh1[:,1], c='r', marker='*')
        # # plt.scatter(muX, muY, c='b')

        # # plt.show()
        # pdfNew = []
        # Pxs = []
        # Pys = []
        # for i in range(len(indices)):
        #     pdfNew.append(cand_pdf[indices[i]])
        #     Pxs.append(candidate_samples[indices[i],0])
        #     Pys.append(candidate_samples[indices[i],1])
        # pdfNew = np.asarray(pdfNew)
        # mesh1 = np.vstack((Pxs, Pys))
        # mesh1 = np.asarray(mesh1).T
        
        # mesh1 = LP.mapPointsBack(muX, muY, mesh, 1, 1)
        
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # ax.scatter(candidate_samples[:,0], candidate_samples[:,1], cand_pdf, c='r', marker='.')

        # integrand = newIntegrand(muX, muY, mesh1, h)
        # testing = np.squeeze(pdfNew)*integrand
        
        integrand = newIntegrand(muX, muY, mesh, h)
        testing = np.squeeze(pdf)*integrand
        # print(muX)
        # print(muY)
        # print(DM.fillDistance(mesh))
        # plt.figure()
        # plt.scatter(mesh[:,0], mesh[:,1], c='r')
        
        value, condNum= QuadratureByInterpolation(mesh, testing, sigmaX, sigmaY, muX, muY, 20)
        
        if condNum > np.exp(1) or value < 0:
            print("Not Using", value)
            newMeshPoints, newVals = getSufficientMesh3(muX, muY, sigmaX, sigmaY, mesh, pdf)
            newMesh = np.vstack((mesh, newMeshPoints))
            newVals = newIntegrand(muX, muY, newMeshPoints, h)*np.squeeze(newVals)
            # newVals = np.zeros(len(newMeshPoints))
            newVals = np.hstack((testing, np.squeeze(newVals)))
            value, condNum= QuadratureByInterpolation(newMesh, newVals, sigmaX, sigmaY, muX, muY, 50)
            print(value)
            # train_samples1, weights1 = getLejaQuadratureRule(sigmaX, sigmaY ,muX,muY)
            # interp = np.asarray(griddata(mesh, pdf, train_samples1, method='cubic', fill_value=0))
            # value = np.dot(weights1,interp)[0]
            # print(value)
            # if value < 0: 
            #     value = 0
                
            
        # print(value)
        newPDF.append(value)
        condNums.append(condNum)
    newPDF = np.asarray(newPDF)
    condNums = np.asarray([condNums]).T
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(mesh[:,0], mesh[:,1], newPDF, c='r', marker='.')
    
    fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(mesh1[:,0], mesh1[:,1], testing, c='k', marker='.')
    return newPDF,condNums, mesh

        
        # pdf = mesh[:,0]**2+mesh[:,1]**2
        # value = QuadratureByInterpola
    

# Test_GetLejaQuadratureRule()
# Test_LejaQuadratureOnLejaPoints()
newPDF,condNums, meshVals = Test_LejaQuadratureLinearizationOnLejaPointsIfAllPointsKnown()
# newPDF, condNum, meshVals= Test_LejaQuadratureLinearizationOnLejaPoints()
