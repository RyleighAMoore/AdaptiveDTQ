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
from Functions import diff



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


"""
allPoints nx2 array of the original point and the neighbors we consider.
returns transformed points so that point is centered at 0,0
"""
def mapPointsTo(mean, allPoints,cov):    
    dx = mean[0]*np.ones((1,len(allPoints))).T
    dy = mean[1]*np.ones((1,len(allPoints))).T
    delta = np.hstack((dx,dy))
    # scaleX = scaleX*np.ones((1,len(allPoints))).T
    # scaleY = scaleY*np.ones((1,len(allPoints))).T
    # scaleVec = np.hstack((scaleX,scaleY))
    vals = np.linalg.inv(np.linalg.cholesky(cov))@(np.asarray(allPoints).T - delta.T)
    return vals.T

def mapPointsBack(mean, allPoints, cov):    
    dx = mean[0]*np.ones((1,len(allPoints))).T
    dy = mean[1]*np.ones((1,len(allPoints))).T
    delta = np.hstack((dx,dy))
    
    # scaleX = scaleX*np.ones((1,len(allPoints))).T
    # scaleY = scaleY*np.ones((1,len(allPoints))).T
    # scaleVec = np.hstack((scaleX,scaleY))
    vals = np.linalg.cholesky(cov)@np.asarray(allPoints).T + delta.T
    return vals.T

def getLejaSetFromPoints(scale, Mesh, numLejaPointsToReturn, poly, Pdf):   
    # sigmaX = np.sqrt(scale.cov[0,0]); sigmaY = np.sqrt(scale.cov[1,1])
    
    candidatesFull = mapPointsTo(scale.mu, Mesh, np.sqrt(scale.cov))
    indices = [np.nan]
    count = 1
    while math.isnan(indices[0]) or count > 4:
        if count >1:
            print("Trying to find Leja points again using more samples")
        candidates, distances, indik = UM.findNearestKPoints(scale.mu[0][0], scale.mu[1][0], candidatesFull,30*int(count*np.ceil(np.max(diff(np.asarray([0,0]))))), getIndices = True)
            
        Px = candidates[0,0]
        Py = candidates[0,1]
        candidates = candidates[1:]
        
        lejaPointsFinal, indices = getLejaPoints(numLejaPointsToReturn, np.asarray([[Px,Py]]).T, poly, num_candidate_samples = 0, candidateSampleMesh = candidates.T, returnIndices=True)
        count = count+1
    
    if math.isnan(indices[0]):
        # plt.figure()
        # plt.plot(candidates[:,0], candidates[:,1], '*k', label='mesh', markersize=14)
        # plt.plot(Px, Py, '*r',label='Main Point',markersize=14)
        # m = mapPointsTo(Px, Py, Mesh, 1/sigmaX, 1/sigmaY)
        # plt.plot(m[:,0], m[:,1], '.c', label='all',markersize=10)
        # plt.legend()
        # plt.show()
        return 0, 0, indices
    lejaPointsFinal = mapPointsBack(candidates[0],lejaPointsFinal, scale.cov)

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


