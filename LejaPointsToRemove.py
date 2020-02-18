# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 11:21:05 2020

@author: Ryleigh
"""
import pyapprox

from functools import partial
from pyapprox.polynomial_sampling import *
from pyapprox.multivariate_polynomials import PolynomialChaosExpansion, \
    define_poly_options_from_variable_transformation
from pyapprox.variable_transformations import \
     define_iid_random_variable_transformation, RosenblattTransformation, \
     TransformationComposition 
from pyapprox.indexing import compute_hyperbolic_indices, get_total_degree
from pyapprox.univariate_quadrature import gauss_jacobi_pts_wts_1D,\
    clenshaw_curtis_pts_wts_1D
from pyapprox.models.genz import GenzFunction
from scipy.stats import beta as beta, uniform, norm
from pyapprox.density import tensor_product_pdf
#from pyapprox.configure_plots import *
from pyapprox.utilities import get_tensor_product_quadrature_rule
from pyapprox.tests.test_rosenblatt_transformation import rosenblatt_example_2d
import matplotlib.pyplot as plt
import UnorderedMesh as UM
import numpy as np
import Functions  as fun
plt.rcParams['text.usetex'] = False
import GenerateLejaPoints as LP
import pickle      
import scipy as sp



def getLejaPointsForRemoval(num_leja_samples, initial_samples, Mesh, numBasis, dimensions=2):
    num_vars=2
    degree=numBasis
    
    poly = PolynomialChaosExpansion()
    var_trans = define_iid_random_variable_transformation(
        norm(loc=0,scale=1),num_vars)
    
    opts = define_poly_options_from_variable_transformation(var_trans)
    poly.configure(opts)
    indices = compute_hyperbolic_indices(num_vars,degree,1.0)
#    indices = get_total_degree(num_vars, num_points)

    poly.set_indices(indices)
    
    degree = np.sqrt(2*num_leja_samples)
    candidate_samples = Mesh
    
    num_initial_samples = len(initial_samples.T)
    precond_func = lambda matrix, samples: christoffel_weights(matrix)
    
    # basis_matrix = poly.canonical_basis_matrix(candidate_samples)
    # P,L,U = sp.linalg.lu(basis_matrix)
    # PivotList = np.ones((2,np.size(Mesh,1)))
    # for i in range(len(P)):
    #     result = np.where(P[i,:] == 1)[0]
    #     PivotList[0,i] = Mesh.T[result,0]
    #     PivotList[1,i] = Mesh.T[result,1]
    
    samples, data_structures, successBool = get_lu_leja_samples(
        poly.canonical_basis_matrix,
        candidate_samples,num_leja_samples,
        preconditioning_function=precond_func,
        initial_samples=initial_samples)
    
    return np.asarray(samples).T, data_structures[2]

def getLejaPointsToRemove(Px, Py, numNeighbors, mesh, scaleX, scaleY, numBasis):
    neighbors, distances = UM.findNearestKPoints(Px, Py, mesh, numNeighbors-1) 
    if len(neighbors > 0): 
        neighbors = np.vstack((neighbors,[Px,Py]))
    else: # make sure we have at least one point.
        neighbors = np.asarray([[Px],[Py]]).T
        
    intialPoints = LP.mapPointsTo(Px,Py,neighbors, 1/scaleX,1/scaleY)
    result = None
    tries = 0
    lejaPointsFinal = None
    while lejaPointsFinal is None:
        try:
            # connect
            lejaPointsFinal, indices = getLejaPointsForRemoval(int(np.ceil(len(mesh))), intialPoints.T, mesh.T, 15+tries, dimensions=2)
        except:
            tries = tries+10
            pass
    
    lejaPointsFinal = LP.mapPointsBack(Px,Py,lejaPointsFinal, scaleX, scaleY)
    # newLeja = LP.mapPointsBack(Px,Py,newLeja,scaleX,scaleY)
    plot= False
    if plot:
        plt.figure()
        plt.plot(neighbors[:,0], neighbors[:,1], '*k', label='Neighbors', markersize=14)
        plt.plot(Px, Py, '*r',label='Main Point',markersize=14)
        plt.plot(lejaPointsFinal[:,0], lejaPointsFinal[:,1], '.c', label='Leja Points',markersize=10)
        plt.legend()
        plt.show()
            
    return lejaPointsFinal, indices


def getMeshIndicesToRemoveFromMesh(mesh, skipCount):  
    initial_samples = np.asarray([[mesh[0,0]], [mesh[0,0]]])
    try:
        LPVals, indices = getLejaPointsToRemove(mesh[0,0], mesh[0,1], len(mesh), mesh, 0.1, 0.1, 55)
    except:
        LPVals, indices = getLejaPointsToRemove(mesh[0,0], mesh[0,1], len(mesh), mesh, 0.1, 0.1, len(mesh))

    valsToKeep = np.ndarray.tolist(LPVals)
    meshList = np.ndarray.tolist(mesh)
    plt.figure()
    plt.plot(LPVals[::skipCount][:,0], LPVals[::skipCount][:,1], '.r', label='Points to remove',markersize=20)
    # plt.plot(LPVals[:20][:,0], LPVals[:2][:,1], '.r', label='Points to keep',markersize=20)

    plt.plot(LPVals[:,0], LPVals[:,1], '*', label='All Points',markersize=10)
    i=0
    while i < len(indices):
        plt.plot(LPVals[i,0], LPVals[i,1], '.w',markersize=5)
        i+=skipCount
    plt.legend()
    plt.show()
    indicesToRemove = indices[1::skipCount]
    return indicesToRemove



# mesh = LP.generateLejaMesh(0, 0.1,0.1, 100)
pickle_in = open("C:/Users/Rylei/Documents/SimpleDTQ-LejaMesh.p","rb")
mesh = pickle.load(pickle_in)
for val in range(len(mesh)-1,-1,-1):
    xx = mesh[val,0]
    yy = mesh[val,1]
    rad = xx**2 +yy**2 
    if rad < 0.1:
        mesh = np.delete(mesh, val, 0)
        
        
        
# diff = np.ones((len(mesh),2))
# diff[:,0]= 3*np.ones((len(mesh)))
# diff[:,1]= 1*np.ones((len(mesh)))
# mesh = np.vstack((mesh,mesh+ diff))

# index = getMeshIndicesToRemoveFromMesh(mesh,2)

# initial_samples = np.asarray([[mesh[0,0]], [mesh[0,0]]])
# 
# LPVals = getLejaPointsToRemove(mesh[0,0], mesh[0,1], len(mesh), mesh, 0.1, 0.1, 35)
# LP2, LPNew2 = getLejaPointsToRemove(5, 5, len(mesh), mesh, 0.3, 0.1, 0.1, 35)


# plt.figure()
# plt.plot(LPVals[::2][:,0], LPVals[::2][:,1], '.r', label='Points to keep',markersize=20)

# plt.plot(LPVals[:,0], LPVals[:,1], '*', label='All Points',markersize=10)

# # plt.plot(mesh[0,0], mesh[0,1], '.r',markersize=10)

# plt.legend()
# plt.show()

# #lejaPoints = generateLejaMesh(10)



