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
from pyapprox.indexing import compute_hyperbolic_indices
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


def getLejaPointsUniform(num_leja_samples, initial_samples, num_candidate_samples =10000, dimensions=2):
    num_vars=2
    degree=15
    
    poly = PolynomialChaosExpansion()
    var_trans = define_iid_random_variable_transformation(
        uniform(),num_vars)
    opts = define_poly_options_from_variable_transformation(var_trans)
    poly.configure(opts)
    indices = compute_hyperbolic_indices(num_vars,degree,1.0)
    poly.set_indices(indices)
    
    # candidates must be generated in canonical PCE space
    num_candidate_samples = 10000
    generate_candidate_samples=lambda n: np.cos(
        np.random.uniform(0.,np.pi,(num_vars,n)))
    
    # enforcing lu interpolation to interpolate a set of initial points
    # before selecting best samples from candidates can cause ill conditioning
    # to avoid this issue build a leja sequence and use this as initial
    # samples and then recompute sequence with different candidates
    
    # must use canonical_basis_matrix to generate basis matrix
    
    num_initial_samples = len(initial_samples.T)
    
    precond_func = lambda matrix, samples: christoffel_weights(matrix)
    
    #initial_samples, data_structures = get_lu_leja_samples(
    #    poly.canonical_basis_matrix,generate_candidate_samples,
    #    num_candidate_samples,num_initial_samples,
    #    preconditioning_function=precond_func,
    #    initial_samples=initial_samples)

    samples, data_structures = get_lu_leja_samples(
    poly.canonical_basis_matrix,generate_candidate_samples,
    num_candidate_samples,num_leja_samples,
    preconditioning_function=precond_func,
    initial_samples=initial_samples)


    samples = var_trans.map_from_canonical_space(samples)
    plot = True
    if plot: 
        plt.plot(samples.T[:,0], samples.T[:,1], '*r')
        plt.plot(initial_samples.T[:,0], initial_samples.T[:,1], '.k')
        plt.show()
        
    return samples[:, num_initial_samples:]



def getLejaPoints(num_leja_samples, initial_samples, num_candidate_samples =10000, dimensions=2):
    num_vars=2
    degree=15
    
    poly = PolynomialChaosExpansion()
    var_trans = define_iid_random_variable_transformation(
        norm(loc=0,scale=0.1),num_vars)
#    var_transY = define_iid_random_variable_transformation(
#        norm(loc=initial_samples[1,0],scale=0.1),num_vars)
    opts = define_poly_options_from_variable_transformation(var_trans)
    poly.configure(opts)
    indices = compute_hyperbolic_indices(num_vars,degree,1.0)
    poly.set_indices(indices)
    
    degree = np.sqrt(2*num_leja_samples)
    generate_candidate_samples = lambda n: np.sqrt(2*degree)*np.random.normal(0, 1, (num_vars, n))
    candidate_samples = generate_candidate_samples(num_candidate_samples)  
    
#    plt.scatter(candidate_samples[0,:], candidate_samples[1,:])
#    plt.show()
    # enforcing lu interpolation to interpolate a set of initial points
    # before selecting best samples from candidates can cause ill conditioning
    # to avoid this issue build a leja sequence and use this as initial
    # samples and then recompute sequence with different candidates
    
    # must use canonical_basis_matrix to generate basis matrix
    #if initial_samples == None:
    #num_initial_samples = 10
    #else:
    initial_samples = var_trans.map_to_canonical_space(initial_samples)
    Px = initial_samples[0,0]
    Py= initial_samples[1,0]
    initial_samples = mapPointsTo(Px, Py, initial_samples.T).T
    
    
    num_initial_samples = float(len(initial_samples.T))
    precond_func = lambda matrix, samples: christoffel_weights(matrix)
#    initial_samples, data_structures = get_lu_leja_samples(
#        poly.canonical_basis_matrix,generate_candidate_samples,
#        num_candidate_samples,num_initial_samples,
#        preconditioning_function=precond_func,
#        initial_samples=initial_samples)
#    
    samples, data_structures = get_lu_leja_samples(
        poly.canonical_basis_matrix,generate_candidate_samples,
        num_candidate_samples,num_leja_samples,
        preconditioning_function=precond_func,
        initial_samples=initial_samples)
    
    samples = var_trans.map_from_canonical_space(samples)

#    plot = True
#    if plot: 
#        plt.plot(samples.T[:,0], samples.T[:,1], '*r')
#        plt.plot(initial_samples.T[:,0], initial_samples.T[:,1], '.k')
#        plt.show()
        
    return samples
    return samples[:, num_initial_samples:]

#initial = np.asarray([[0,0], [-.2,0], [0,.2]]).T
#lejaPoints = getLejaPoints(8, initial)

#neighbors = UM.findNearestKPoints(Px, Py, allPoints, 4)
"""
point 1x2 array
allPoints nx2 array of the original point and the neighbors we consider.
returns transformed points so that point is centered at 0,0
"""
def mapPointsTo(Px, Py, allPoints):    
    dx = Px*np.ones((1,len(allPoints))).T
    dy = Py*np.ones((1,len(allPoints))).T
    delta = np.hstack((dx,dy))
    return np.asarray(allPoints) - delta

def mapPointsBack(Px, Py, allPoints):    
    dx = Px*np.ones((1,len(allPoints))).T
    dy = Py*np.ones((1,len(allPoints))).T
    delta = np.hstack((dx,dy))
    return np.asarray(allPoints) + delta


mesh = UM.generateOrderedGridCenteredAtZero(-1.8, 1.8, -1.8, 1.8, 0.1)      # ordered mesh 
#mesh = UM.generateRandomPoints(-0.2,0.2,-0.2,0.2,100000)  # unordered mesh

num = 435
point = np.asarray(mesh[num:num+1,:])
Px = point[0,0]
Py= point[0,1]
neighbors = UM.findNearestKPoints(Px, Py, mesh, 6)
#val = mapPointsTo(Px, Py, neighbors)
lejaPoints = getLejaPoints(10, neighbors.T)

#lejaPoints = getLejaPointsUniform(10, None)

lejaPointsFinal = mapPointsBack(Px, Py, lejaPoints.T)

plt.figure()
plt.plot(neighbors[:,0], neighbors[:,1], '*k', label='Neighbors', markersize=14)
plt.plot(Px, Py, '*r',label='Main Point',markersize=14)
plt.plot(lejaPointsFinal[:,0], lejaPointsFinal[:,1], '.c', label='Leja Points',markersize=10)

plt.legend()
plt.show()


#

