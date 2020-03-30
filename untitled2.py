# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 22:26:30 2020

@author: Rylei
"""

import chaospy
import pyapprox
import GenerateLejaPoints as LP
import numpy as np
from pyapprox.variables import IndependentMultivariateRandomVariable
from pyapprox.variable_transformations import \
AffineRandomVariableTransformation
from pyapprox.multivariate_polynomials import PolynomialChaosExpansion,\
define_poly_options_from_variable_transformation
from pyapprox.probability_measure_sampling import \
generate_independent_random_samples
from scipy.stats import uniform, beta, norm
from pyapprox.indexing import compute_hyperbolic_indices, tensor_product_indices,compute_tensor_product_level_indices
from pyapprox.models.genz import GenzFunction
from functools import partial
from pyapprox.univariate_quadrature import gauss_jacobi_pts_wts_1D, \
clenshaw_curtis_in_polynomial_order
from pyapprox.utilities import get_tensor_product_quadrature_rule
from pyapprox.polynomial_sampling import christoffel_weights
import numpy as np
from scipy.stats import multivariate_normal
from pyapprox.multivariate_polynomials import evaluate_multivariate_orthonormal_polynomial
import UnorderedMesh as UM
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Functions import f1, f2, g1, g2
import getPCE as PCE

distribution = chaospy.J(chaospy.Normal(0, .1), chaospy.Normal(0, .1))
polyChaos = chaospy.orth_ttr(15, distribution)
distribution = chaospy.J(chaospy.Normal(0, 1), chaospy.Normal(0, .1))

# poly = chaospy.orth_ttr(15, distribution)
  
def QuadratureByInterpolation(train_samples, train_values, sigmaX, sigmaY, muX, muY, degree):
    poly = PCE.generatePCE(degree, muX=muX, muY=muY, sigmaX=sigmaX, sigmaY=sigmaY)
    '''
    univariate_variables = [norm(muX,sigmaX),norm(muY,sigmaY)]
    variable = IndependentMultivariateRandomVariable(univariate_variables)
    var_trans = AffineRandomVariableTransformation(variable)
    
    poly = PolynomialChaosExpansion()
    poly_opts = define_poly_options_from_variable_transformation(var_trans)
    poly.configure(poly_opts)
    degree=degree
    num_vars = 2
    # indices = compute_hyperbolic_indices(poly.num_vars(),degree,1.0)
    # indices = compute_tensor_product_level_indices(poly.num_vars(),degree,max_norm=True)
    # degrees = [int(train_samples.shape[0]**(1/poly.num_vars()))]*poly.num_vars()
    # indices = tensor_product_indices(degrees)
    poly.set_indices(indices)
    '''

    # train_values = np.log(train_values)
    basis_matrix = poly.basis_matrix(train_samples.T)
    basis_matrix = polyChaos(train_samples[:,0], train_samples[:,1]).T
    numRows = np.size(basis_matrix,0)
    numCols = np.size(basis_matrix,1)
    ZeroCoeffs = np.zeros((numCols-numRows,1))
    assert len(ZeroCoeffs) >= 0
    
    basis_matrix = basis_matrix[:,:numRows]
    
    Vinv = np.linalg.inv(basis_matrix)
    # print("cond=",np.sum(np.abs(Vinv[0,:])))
    
    # precond_weights = christoffel_weights(basis_matrix)
    # precond_basis_matrix = precond_weights[:,np.newaxis]*basis_matrix
    # precond_train_values = precond_weights[:,np.newaxis]*train_values

    assert np.size(basis_matrix,0) == np.size(basis_matrix,1)
    # coef = np.linalg.lstsq(basis_matrix,train_values)[0]
    coef = np.matmul(Vinv, train_values)
    # print(np.max(coef-coef2), '---------------')
    # assert np.max(np.abs(coef-coef2)) <0.01 
    # coef = np.matmul(train_values, Vinv)
    coef = np.asarray([coef]).T
    coef = np.vstack((coef, ZeroCoeffs))

    # if np.sum(np.abs(Vinv[0,:])) <-1:
    # poly.set_coefficients(coef)
    
    # indices = poly.indices
    # recursion_coeffs = np.asarray(poly.recursion_coeffs)
    
    # mesh = UM.generateOrderedGridCenteredAtZero(-0.4,0.4, -0.4,0.4, 0.01, includeOrigin=True)
    # mesh = LP.generateLejaMeshNotCentered(200, min(sigmaX, sigmaY), min(sigmaY,sigmaY), 40, muX, muY)
    # vals1 =(poly.value((mesh).T))
    # vals2 =(poly.value((train_samples).T))
    # print('condNum', np.sum(np.abs(Vinv[0,:])))
    # print(np.max((np.squeeze(vals2,-1)-train_values)), '*****')

    # rv = multivariate_normal([0, 0], [[np.sqrt(0.01)*g1(), 0], [0, np.sqrt(0.01)*g1()]])
    # pdfNorm = np.asarray([rv.pdf(mesh)]).T
    
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(mesh[:,0], mesh[:,1], vals1, c='r', marker='.')
    # ax.scatter(train_samples[:,0], train_samples[:,1], train_values, c='k', marker='o')
    # ax.scatter(mesh[:,0], mesh[:,1], pdfNorm, c='k', marker='.')

    # ax.set_zlim(0, 21)

    # print(np.max(train_values-vals1))
    # print(coef[0])
    # if coef[0][0] < 0:
    #     print("returned 0")
    #     return [0], poly
    # if coef[0][0]-1 >0.02:
    #     fig = plt.figure()
    #     ax = Axes3D(fig)
    #     ax.scatter(train_samples[:,0], train_samples[:,1], vals1, c='r', marker='.')
    #     ax.scatter(train_samples[:,0], train_samples[:,1], train_values, c='k', marker='.')
        
    
    return coef[0], np.sum(np.abs(Vinv[0,:]))


poly = PCE.generatePCE(31)
mesh, mesh2 = LP.getLejaPointsWithStartingPoints([0,0,.1,.1], 100, 5000, poly)
# mesh2 , mesh3 = LP.getLejaPointsWithStartingPoints([0,0,sqrt(h)*fun.g1()*1.5,sqrt(h)*fun.g2()*1.5], 230, 5000, poly)
# mesh3 = checkDist(mesh, mesh3, 0.03)
# mesh = np.vstack((mesh,mesh3))

# poly = PCE.generatePCE(2)
# vals = np.sum(poly.basis_matrix(mesh))
# polyChaos = chaospy.orth_ttr(2, distribution)
# distribution = chaospy.J(chaospy.Normal(0, .1), chaospy.Normal(0, .1))

# initial_samples = np.asarray([[0,0]]).T
# poly = PCE.generatePCE_Uniform(50)
# lejas, new = LP.getLejaPoints_Uniform(530, initial_samples, poly, num_candidate_samples = 2000, candidateSampleMesh = [], returnIndices = False)

plt.scatter(mesh[:,0], mesh[:,1])

pdf = UM.generateICPDF(mesh[:,0], mesh[:,1], .1, .1)


aa, cond = QuadratureByInterpolation(mesh, pdf, .1, .1, 0, 0, 20)




