# # -*- coding: utf-8 -*-
# """
# Created on Tue Jan 21 11:21:05 2020

# @author: Ryleigh
# """
# import pyapprox

# from functools import partial
# from pyapprox.polynomial_sampling import *
# from pyapprox.multivariate_polynomials import PolynomialChaosExpansion, \
#     define_poly_options_from_variable_transformation
# from pyapprox.variable_transformations import \
#      define_iid_random_variable_transformation, RosenblattTransformation, \
#      TransformationComposition 
# from pyapprox.indexing import compute_hyperbolic_indices, get_total_degree
# from pyapprox.univariate_quadrature import gauss_jacobi_pts_wts_1D,\
#     clenshaw_curtis_pts_wts_1D
# from pyapprox.models.genz import GenzFunction
# from scipy.stats import beta as beta, uniform, norm
# from pyapprox.density import tensor_product_pdf
# #from pyapprox.configure_plots import *
# from pyapprox.utilities import get_tensor_product_quadrature_rule
# from pyapprox.tests.test_rosenblatt_transformation import rosenblatt_example_2d
# import matplotlib.pyplot as plt
# import UnorderedMesh as UM
# import numpy as np
# import Functions  as fun
# plt.rcParams['text.usetex'] = False
# import matplotlib.animation as animation
# import pickle
# from pyapprox.variables import IndependentMultivariateRandomVariable
# from pyapprox.variable_transformations import \
# AffineRandomVariableTransformation
# import getPCE as PCE
# import pickle


# # generate_candidate_samples = lambda n: np.sqrt(2*np.sqrt(2*230))*np.random.normal(0, 1, (2, n))
# # cs = generate_candidate_samples(5000)
# # pkl_file = open("C:/Users/Rylei/Documents/SimpleDTQ/PickledData/cs.p", "rb" ) 
# # cs = pickle.load(pkl_file)

# '''
# num_leja_samples: Total number of samples to be returned (including initial samples).
# initial_samples: The samples that we must include in the leja sequence.
# poly: Polynomial chaos expansion, fully implemented with options, indices, etc.
# num_candidate_samples: Number of samples that are chosen from to complete the leja sequence after initial samples are used.
# dimensions: number of dimensions in the problem
# candidateSampleMesh: If num_candidate_samples is zero, this variable defines the candidate samples to use
# returnIndices: Returns the indices of the leja sequence if True.
# '''
# def getLejaPoints(num_leja_samples, initial_samples, poly, num_candidate_samples = 5000, candidateSampleMesh = [], returnIndices = False):        
#     # print(num_candidate_samples)
#     assert poly.var_trans.scale_parameters[0][0] == 0.0, "The polynomial should have standard mean"
#     assert poly.var_trans.scale_parameters[0][1] == 1.0, "The polynomial should have standard variance"
#     assert num_leja_samples <= len(poly.indices.T)
#     num_vars = poly.num_vars()

#     generate_candidate_samples = lambda n: np.sqrt(2*np.sqrt(2*num_leja_samples))*np.random.normal(0, 1, (num_vars, n)) 
#     # generate_candidate_samples = lambda n: 4*np.random.normal(0, 1, (num_vars, n)) 
#     # generate_candidate_samples = lambda n: 10*np.random.normal(0, 1, (num_vars, n)) 

#     if num_candidate_samples == 0:
#         candidate_samples = candidateSampleMesh
#     else:
#         candidate_samples = generate_candidate_samples(num_candidate_samples)
#         # candidate_samples = cs
#         # plt.scatter(candidate_samples[0,:], candidate_samples[1,:], c='r', marker='.')
# # 
#     num_initial_samples = len(initial_samples.T)
#     precond_func = lambda matrix, samples: christoffel_weights(matrix)
# #    initial_samples, data_structures = get_lu_leja_samples(
# #        poly.canonical_basis_matrix,generate_candidate_samples,
# #        num_candidate_samples,num_initial_samples,
# #        preconditioning_function=precond_func,
# #        initial_samples=initial_samples)
    
#     samples, data_structures, successBool = get_lu_leja_samples(
#         poly.canonical_basis_matrix,
#         candidate_samples,num_leja_samples,
#         preconditioning_function=precond_func,
#         initial_samples=initial_samples)
    
    
#     if returnIndices:
#         assert successBool == True, "Need to implement returning indices when successBool is False."
        
#     if successBool ==True:
#         if returnIndices:
#             indicesLeja = data_structures[2]
#             return np.asarray(samples).T, indicesLeja
#         assert len(np.asarray(samples).T) <= len(poly.indices.T)
#         return np.asarray(samples).T, np.asarray(samples[:,num_initial_samples:]).T
  
#     if successBool == False:
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
#             samples2, data_structures2, successBool = get_lu_leja_samples(poly.canonical_basis_matrix,candidate_samples,num_leja_samples,preconditioning_function=precond_func,initial_samples=initial_samples_edited)
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

#             samples2, data_structures2, successBool = get_lu_leja_samples(poly.canonical_basis_matrix,candidate_samples,num_leja_samples_edited,preconditioning_function=precond_func,initial_samples=initial_samples_edited)
#             if successBool == True:
# #                initial_samples_edited = np.copy(samples2[:,0:num_initial_samples_edited])
#                 numInitialAdded += 1
#                 print("successfully Added a Point")
#             if successBool == False: 
#                 pointsRemoved.append(np.asarray([initial_samples_edited[:,0]]).T)
#                 initial_samples_edited = np.delete(initial_samples_edited,0,1)
#                 num_leja_samples_edited = len(initial_samples_edited[1,:])+1  # Want to add one leja point
#                 samples, data_structures, successBool = get_lu_leja_samples(poly.canonical_basis_matrix,candidate_samples,num_leja_samples_edited,preconditioning_function=precond_func,initial_samples=initial_samples_edited)                
#                 initial_samples_edited = np.copy(samples[0:len(initial_samples_edited[1,:])+1,:])
#                 newLejaSamples.append(np.copy(initial_samples_edited[:,-1]))
        
#         for i in range(num_initial_samples_edited, len(samples2[1, :])):    
#             newLejaSamples.append(np.asarray(samples2[:, i]))
#     samples = samples2[:, :num_leja_samples]
#     assert len(np.asarray(samples).T) <= len(poly.indices.T)
#     return np.asarray(samples).T, np.asarray(newLejaSamples)


# # expansion1 = chaospy.orth_ttr(2, chaospy.Normal(1, .1))
# # distribution = chaospy.J(chaospy.Normal(0, 1), chaospy.Normal(0, 1))
# # poly = chaospy.orth_ttr(10, distribution)
# # one, two = getLejaPoints(20, np.asarray([[0,0]]).T, poly, candidateSampleMesh = [], returnIndices = False)
# # # plt.figure()
# # plt.scatter(one[:,0], one[:,1])

# """
# allPoints nx2 array of the original point and the neighbors we consider.
# returns transformed points so that point is centered at 0,0
# """
# def mapPointsTo(Px, Py, allPoints,scaleX, scaleY):    
#     dx = Px*np.ones((1,len(allPoints))).T
#     dy = Py*np.ones((1,len(allPoints))).T
#     delta = np.hstack((dx,dy))
#     scaleX = scaleX*np.ones((1,len(allPoints))).T
#     scaleY = scaleY*np.ones((1,len(allPoints))).T
#     scaleVec = np.hstack((scaleX,scaleY))
#     vals = (np.asarray(allPoints) - delta)*scaleVec
#     return vals

# def mapPointsBack(Px, Py, allPoints, scaleX, scaleY):    
#     dx = Px*np.ones((1,len(allPoints))).T
#     dy = Py*np.ones((1,len(allPoints))).T
#     delta = np.hstack((dx,dy))
#     scaleX = scaleX*np.ones((1,len(allPoints))).T
#     scaleY = scaleY*np.ones((1,len(allPoints))).T
#     scaleVec = np.hstack((scaleX,scaleY))
#     vals = (scaleVec)*np.asarray(allPoints) + delta
#     return vals

# '''
# scaleParams = [muX, muY, sigmaX, sigmaY]
# numNewLejaPoints = number of desired points
# numCandidateSamples = number of samples to chose candidates from.
# poly: standard normal PCE
# neighbors = [numNeighbors, mesh]
# '''
# def getLejaPointsWithStartingPoints(scaleParams, numLejaPoints, numCandidateSamples, poly, neighbors=[0,[]]):
#     Px = scaleParams[0]; Py = scaleParams[1]
#     sigmaX = scaleParams[2]; sigmaY = scaleParams[3]
#     if neighbors[0] > 0: 
#         numNeighbors = neighbors[0]; mesh = neighbors[1]
#         neighbors, distances = UM.findNearestKPoints(Px, Py, mesh, numNeighbors) 
#         neighbors = np.vstack((neighbors,[Px,Py]))
#     else: # make sure we have at least one point.
#         numNeighbors = 0
#         neighbors = np.asarray([[Px],[Py]]).T
        
#     intialPoints = mapPointsTo(Px,Py,neighbors, 1/sigmaX,1/sigmaY)
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

# # poly = PCE.generatePCE(20, muX=0, muY=0, sigmaX = 1, sigmaY=1)
# # one, mesh2 = getLejaPointsWithStartingPoints([0,0,.1,.1], 230, 1000, poly)
# # mesh, mesh2 = getLejaPointsWithStartingPoints([0,0,.1,.1], 12, 1000, poly, neighbors=[3,one])


# def getLejaSetFromPoints(scaleParams, mesh, numNewLejaPoints, poly):
#     assert numNewLejaPoints <= np.size(mesh,0), "Asked for subset is bigger than whole set"
#     Px = scaleParams[0]; Py = scaleParams[1]
#     sigmaX = scaleParams[2]; sigmaY = scaleParams[3]
    
#     candidates = mapPointsTo(Px, Py, mesh, 1/sigmaX, 1/sigmaY)
#     lejaPointsFinal, indices = getLejaPoints(numNewLejaPoints, np.asarray([[0,0]]).T, poly, num_candidate_samples = 0, candidateSampleMesh = candidates.T, returnIndices=True)
#     lejaPointsFinal = mapPointsBack(Px,Py,lejaPointsFinal, sigmaX, sigmaY)
     
#     plot= False
#     if plot:
#         plt.figure()
#         plt.plot(mesh[:,0], mesh[:,1], '*k', label='mesh', markersize=14)
#         plt.plot(Px, Py, '*r',label='Main Point',markersize=14)
#         plt.plot(lejaPointsFinal[:,0], lejaPointsFinal[:,1], '.c', label='Leja Points',markersize=10)
#         plt.legend()
#         plt.show()
#     lejaPointsFinal
#     return lejaPointsFinal, indices


# # one, two = getLejaPointsWithStartingPoints([0,0,.5,.5], 230, 1000, poly)
# # mesh, mesh2 = getLejaSetFromPoints([0,0,.1,.1], one, 100, poly)


# # num_leja_samples = 230
# # num_vars=2
# # generate_candidate_samples = lambda n: np.sqrt(2*np.sqrt(2*num_leja_samples))*np.random.normal(0, 1, (num_vars, n)) 
# # candidate_samples = generate_candidate_samples(5000)
# # poly = PCE.generatePCE(20, muX=0, muY=0, sigmaX = 1, sigmaY=1)
# # one, mesh2 = getLejaPointsWithStartingPoints([0,0,1,1], 231, 5000, poly)

# # poly2 = PCE.generatePCE(30, muX=0, muY=0, sigmaX = 1, sigmaY=1)
# # one2, mesh22 = getLejaPointsWithStartingPoints([0,0,.1,.1], 231, 5000, poly2)

# # plt.figure()
# # plt.scatter(one[:,0], one[:,1], label='231 points picked by deg 20 poly')
# # plt.scatter(one2[:,0], one2[:,1], marker='.', label='231 points picked by deg 30 poly')
# # plt.legend()