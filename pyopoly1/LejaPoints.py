# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 11:21:05 2020

@author: Ryleigh
"""
import matplotlib.pyplot as plt
import numpy as np
import UnorderedMesh as UM
from pyopoly1 import opolynd
from pyopoly1.LejaUtilities import get_lu_leja_samples, sqrtNormal_weights
from pyopoly1.opolynd import opolynd_eval
import UnorderedMesh as UM
from mpl_toolkits.mplot3d import Axes3D
import math



'''
num_leja_samples: Total number of samples to be returned (including initial samples).
initial_samples: The samples that we must include in the leja sequence.
poly: Polynomial chaos expansion, fully implemented with options, indices, etc.
num_candidate_samples: Number of samples that are chosen from to complete the leja sequence after initial samples are used.
candidateSampleMesh: If num_candidate_samples is zero, this variable defines the candidate samples to use
returnIndices: Returns the indices of the leja sequence if True.
'''
def getLejaPoints(num_leja_samples, initial_samples, poly, num_candidate_samples = 10000, candidateSampleMesh = [], returnIndices = False):
    num_vars = np.size(initial_samples,0)
    # generate_candidate_samples = lambda n: np.sqrt(2*np.sqrt(2*num_leja_samples))*np.random.normal(0, 1, (num_vars, n)) 
    generate_candidate_samples = lambda n: 7*np.random.normal(0, 1, (num_vars, n)) 
    # generate_candidate_samples = lambda n: np.sqrt(2)*num_leja_samples*np.random.normal(0, 1, (num_vars, n)) 


    if num_candidate_samples == 0:
        candidate_samples = candidateSampleMesh
    else:
        candidate_samples = generate_candidate_samples(num_candidate_samples)

    num_initial_samples = len(initial_samples.T)
    
    # precond_func = lambda matrix, samples: christoffel_weights(matrix)
    precond_func = lambda matrix, samples: sqrtNormal_weights(samples)
    
    samples, data_structures, successBool = get_lu_leja_samples(poly,
        opolynd_eval,candidate_samples,num_leja_samples,
        preconditioning_function=precond_func,
        initial_samples=initial_samples)
    
    
    if returnIndices:
        if successBool == False:
            print("LEJA FAIL - LEJA FAIL - LEJA FAIL")
            return [float('nan')], [float('nan')]
        assert successBool == True, "Need to implement returning indices when successBool is False."
        
    if successBool ==True:
        if returnIndices:
            indicesLeja = data_structures[2]
            return np.asarray(samples).T, indicesLeja
        return np.asarray(samples).T, np.asarray(samples[:,num_initial_samples:]).T
  
#     if successBool == False:
#         print("Long Leja")
#         numInitialAdded = 0
#         pointsRemoved = []
#         initial_samples_edited = np.copy(initial_samples)
#         newLejaSamples = []
#         ii=0
#         while successBool == False: # Truncate initalSamples until succed to add a Leja point
#             print("Truncating Initial Samples")
#             assert len(pointsRemoved) <= num_initial_samples, "Removed all Initial points"
#             pointsRemoved.append(np.asarray([initial_samples_edited[:,0]]).T)
#             initial_samples_edited = np.delete(initial_samples_edited,0,1)
#             num_initial_samples_edited = len(initial_samples_edited.T) 
#             samples2, data_structures2, successBool = get_lu_leja_samples(poly, opolynd_eval,candidate_samples,num_leja_samples,preconditioning_function=precond_func,initial_samples=initial_samples_edited)
#             ii+=1
#         initial_samples_edited = np.copy(samples2[:, 0:num_initial_samples_edited+1])
#         numInitialAdded = num_initial_samples - ii# Able to add a Leja point!
#         newLejaSamples.append(np.copy(initial_samples_edited[:,-1]))
#         ii=0
#         while len(pointsRemoved) != 0: #Try to add one more point in Leja sequence
#             print("Trying to Add a point")
#             ii+=1
#             pointToAdd = pointsRemoved.pop(-1)
#             initial_samples_edited = np.hstack((pointToAdd,initial_samples_edited))
#             num_initial_samples_edited = len(initial_samples_edited[1,:])
#             num_leja_samples_edited = len(initial_samples_edited[1,:]) # Want to try and add the points
#             num_leja_samples_edited = num_leja_samples # Want to try and add the points

#             samples2, data_structures2, successBool = get_lu_leja_samples(poly,
#         opolynd_eval,candidate_samples,num_leja_samples_edited,preconditioning_function=precond_func,initial_samples=initial_samples_edited)
#             if successBool == True:
# #                initial_samples_edited = np.copy(samples2[:,0:num_initial_samples_edited])
#                 numInitialAdded += 1
#                 print("successfully Added a Point")
#             if successBool == False: 
#                 pointsRemoved.append(np.asarray([initial_samples_edited[:,0]]).T)
#                 initial_samples_edited = np.delete(initial_samples_edited,0,1)
#                 num_leja_samples_edited = len(initial_samples_edited[1,:])+1  # Want to add one leja point
#                 samples, data_structures, successBool = get_lu_leja_samples(poly,
#         opolynd_eval,candidate_samples,num_leja_samples_edited,preconditioning_function=precond_func,initial_samples=initial_samples_edited)                
#                 initial_samples_edited = np.copy(samples[0:len(initial_samples_edited[1,:])+1,:])
#                 newLejaSamples.append(np.copy(initial_samples_edited[:,-1]))
        
#         for i in range(num_initial_samples_edited, len(samples2[1, :])):    
#             newLejaSamples.append(np.asarray(samples2[:, i]))
#     samples = samples2[:, :num_leja_samples]
#     assert len(np.asarray(samples).T) <= len(poly.indices.T)
#     return np.asarray(samples).T, np.asarray(newLejaSamples)


'''Some code for testing - Should make a test file out of some of these'''
# from families import HermitePolynomials
# import indexing
# H = HermitePolynomials(rho=0)
# d=2
# k = 20   
# ab = H.recurrence(k+1)
# lambdas = indexing.total_degree_indices(d, k)
# H.lambdas = lambdas
# one, two = getLejaPoints(231, np.asarray([[0,0]]).T, H, candidateSampleMesh = [], returnIndices = False)
# plt.figure()
# plt.scatter(one[:,0], one[:,1])

"""
allPoints nx2 array of the original point and the neighbors we consider.
returns transformed points so that point is centered at 0,0
"""
def mapPointsTo(Px, Py, allPoints,scaleX, scaleY):    
    dx = Px*np.ones((1,len(allPoints))).T
    dy = Py*np.ones((1,len(allPoints))).T
    delta = np.hstack((dx,dy))
    scaleX = scaleX*np.ones((1,len(allPoints))).T
    scaleY = scaleY*np.ones((1,len(allPoints))).T
    scaleVec = np.hstack((scaleX,scaleY))
    vals = (np.asarray(allPoints) - delta)*scaleVec
    return vals

def mapPointsBack(Px, Py, allPoints, scaleX, scaleY):    
    dx = Px*np.ones((1,len(allPoints))).T
    dy = Py*np.ones((1,len(allPoints))).T
    delta = np.hstack((dx,dy))
    scaleX = scaleX*np.ones((1,len(allPoints))).T
    scaleY = scaleY*np.ones((1,len(allPoints))).T
    scaleVec = np.hstack((scaleX,scaleY))
    vals = (scaleVec)*np.asarray(allPoints) + delta
    return vals

'''
scaleParams = [muX, muY, sigmaX, sigmaY]
numNewLejaPoints = number of desired points
numCandidateSamples = number of samples to chose candidates from.
poly: standard normal PCE
neighbors = [numNeighbors, mesh]
'''
# def getLejaPointsWithStartingPoints(scaleParams, numLejaPoints, numCandidateSamples, poly, neighbors=[0,[]]):
#     Px = scaleParams.mu[0][0]; Py = scaleParams.mu[0][0]
#     sigmaX = np.sqrt(scaleParams.cov[0,0]); sigmaY = np.sqrt(scaleParams.cov[1,1])
#     if neighbors[0] > 0: 
#         numNeighbors = neighbors[0]; mesh = neighbors[1]
#         neighbors, distances = UM.findNearestKPoints(Px, Py, mesh, numNeighbors) 
#         neighbors = np.vstack((neighbors,[Px,Py]))
#     else: # make sure we have at least one point.
#         numNeighbors = 0
#         neighbors = np.asarray([[Px],[Py]]).T
        
#     intialPoints = mapPointsTo(Px,Py,neighbors, 1,1)
#     lejaPointsFinal1, newLeja = getLejaPoints(numLejaPoints, intialPoints.T, poly, num_candidate_samples=numCandidateSamples)
#     lejaPointsFinal = mapPointsBack(Px,Py,lejaPointsFinal1, sigmaX, sigmaY)
#     newLeja = mapPointsBack(Px,Py,newLeja,sigmaX,sigmaY)
    
#     plot= False
#     if plot:
#         plt.figure()
#         plt.plot(neighbors[:,0], neighbors[:,1], '*k', label='Neighbors', markersize=14)
#         plt.plot(Px, Py, '*r',label='Main Point',markersize=14)
#         plt.plot(lejaPointsFinal[:,0], lejaPointsFinal[:,1], '.c', label='Leja Points',markersize=10)
#         # plt.plot(lejaPointsFinal1[:,0], lejaPointsFinal1[:,1], '.r', label='Leja Points Unscaled',markersize=10)

#         plt.legend()
#         plt.show()

#     return lejaPointsFinal, newLeja

# poly = PCE.generatePCE(20, muX=0, muY=0, sigmaX = 1, sigmaY=1)
# one, mesh2 = getLejaPointsWithStartingPoints([0,0,.1,.1], 230, 1000, poly)
# mesh, mesh2 = getLejaPointsWithStartingPoints([0,0,.1,.1], 12, 1000, poly, neighbors=[3,one])

def getLejaSetFromPoints(scale, Mesh, numLejaPointsToReturn, poly, Pdf):   
    sigmaX = np.sqrt(scale.cov[0,0]); sigmaY = np.sqrt(scale.cov[1,1])

    candidatesFull = mapPointsTo(scale.mu[0], scale.mu[1], Mesh, 1/sigmaX, 1/sigmaY)
    candidates, distances, indik = UM.findNearestKPoints(scale.mu[0][0], scale.mu[1][0], candidatesFull,50, getIndices = True)
    Px = candidates[0,0]
    Py = candidates[0,1]
    candidates = candidates[1:]
    
    lejaPointsFinal, indices = getLejaPoints(numLejaPointsToReturn, np.asarray([[Px,Py]]).T, poly, num_candidate_samples = 0, candidateSampleMesh = candidates.T, returnIndices=True)
    if math.isnan(indices[0]):
        # plt.figure()
        # plt.plot(candidates[:,0], candidates[:,1], '*k', label='mesh', markersize=14)
        # plt.plot(Px, Py, '*r',label='Main Point',markersize=14)
        # m = mapPointsTo(Px, Py, Mesh, 1/sigmaX, 1/sigmaY)
        # plt.plot(m[:,0], m[:,1], '.c', label='all',markersize=10)
        # plt.legend()
        # plt.show()
        return 0, 0, indices
    lejaPointsFinal = mapPointsBack(Px,Py,lejaPointsFinal, sigmaX, sigmaY)

    plot= False
    if plot:
        plt.figure()
        plt.plot(Mesh[:,0], Mesh[:,1], '*k', label='mesh', markersize=14)
        plt.plot(Px, Py, '*r',label='Main Point',markersize=14)
        plt.plot(lejaPointsFinal[:,0], lejaPointsFinal[:,1], '.c', label='Leja Points',markersize=10)
        plt.legend()
        plt.show()

    indicesNew = indik[indices]
    return Mesh[indicesNew], Pdf[indicesNew], indicesNew


# '''Some code for testing - Should make a test file out of some of these'''
# if __name__ == "__main__":
#     from Scaling import GaussScale
#     from families import HermitePolynomials
#     import indexing
#     import matplotlib.pyplot as plt
#     from mpl_toolkits.mplot3d import Axes3D
#     h=0.01
#     import Functions as fun
#     import ICMeshGenerator as M
#     poly = HermitePolynomials(rho=0)
#     d=2
#     k = 40    
#     ab = poly.recurrence(k+1)
#     lambdas = indexing.total_degree_indices(d, k)
#     poly.lambdas = lambdas
    
#     IC = np.sqrt(0.005)
#     mesh, two = getLejaPoints(230, np.asarray([[0,0]]).T, poly, candidateSampleMesh = [], returnIndices = False)
#     # mesh = mapPointsBack(0, 0, mesh, IC, IC)
#     mesh = M.getICMesh(1,0.1,h)
    
#     meshtest, two = getLejaPoints(12, np.asarray([[0,0]]).T, poly, num_candidate_samples=5000, returnIndices = False)
#     meshtest = mapPointsBack(0, 0, meshtest, IC, IC)

#     newmesh, two = getLejaPoints(12, np.asarray([[0,0]]).T, poly, num_candidate_samples=0, candidateSampleMesh = mesh.T, returnIndices = False)
#     newmesh = mapPointsBack(0, 0, newmesh, IC, IC)


#     pdf = UM.generateICPDF(mesh[:,0], mesh[:,1], IC, IC)
    
#     ii=0
#     scale = GaussScale(2)
#     scale.setMu(np.asarray([[mesh[ii,0],mesh[ii,1]]]).T)
#     S = IC
#     scale.setSigma(np.asarray([S, S]))
#     numLejaPointsToReturn = 12
    
#     meshFull, pdfNew = getLejaSetFromPoints(scale, mesh, numLejaPointsToReturn, poly, pdf)
    
#     grd = UM.generateOrderedGridCenteredAtZero(-.3, .3, -.3, .3, 0.01, includeOrigin=True)
#     gauss2 = fun.Gaussian(scale, grd)
    
#     fig = plt.figure()
#     ax = Axes3D(fig)
#     ax.scatter(grd[:,0], grd[:,1], gauss2, c='b', marker='.')
#     ax.scatter(mesh[:,0], mesh[:,1], pdf, c='k', marker='.')
#     ax.scatter(meshFull[:,0], meshFull[:,1], pdfNew, c='r', marker='o')

#     ax.scatter(mesh[ii,0], mesh[ii,1], np.max(pdf), c='g', marker='o')


#     plt.figure()
#     plt.scatter(meshFull[:,0], meshFull[:,1], label='Chosen points')
#     plt.scatter(meshtest[:,0], meshtest[:,1],c='r', label='Real points')


    
    
    
    
    
    
    
    
    
    
    

