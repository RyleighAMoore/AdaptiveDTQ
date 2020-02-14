
import pyapprox
import unittest
from scipy import special as sp
from pyapprox.multivariate_polynomials import *
from pyapprox.univariate_quadrature import gauss_hermite_pts_wts_1D, \
    gauss_jacobi_pts_wts_1D
from pyapprox.utilities import get_tensor_product_quadrature_rule, approx_fprime
from pyapprox.variable_transformations import \
     define_iid_random_variable_transformation, IdentityTransformation,\
     AffineRandomVariableTransformation
from pyapprox.density import map_to_canonical_gaussian
from pyapprox.variables import IndependentMultivariateRandomVariable,\
    float_rv_discrete
from functools import partial
from pyapprox.indexing import sort_indices_lexiographically
from scipy.stats import uniform, beta, norm, hypergeom, binom
import GenerateLejaPoints as LP
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['text.usetex'] = False
import numpy as np

num_vars = 2
degree = 15
deriv_order=0    
probability_measure = True
num_leja_samples = 100
initial_samples = np.asarray([[0],[0]])

lejaPoints, newLeja = LP.getLejaPoints(num_leja_samples, initial_samples,degree, num_candidate_samples = 5000, dimensions=num_vars)
rv = multivariate_normal([0, 0], [[1, 0], [0, 1]])
pdf = rv.pdf(lejaPoints)

# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(lejaPoints[:,0], lejaPoints[:,1], pdf, c='r', marker='.')

poly = PolynomialChaosExpansion()
var_trans = define_iid_random_variable_transformation(
    norm(0,1),num_vars) 
poly.configure({'poly_type':'hermite','var_trans':var_trans})
indices = compute_hyperbolic_indices(num_vars,degree,1.0)

# sort lexographically to make testing easier
I = np.lexsort((indices[0,:],indices[1,:], indices.sum(axis=0)))
indices = indices[:,I]
poly.set_indices(indices)
basis_matrix = poly.canonical_basis_matrix(lejaPoints.T)
basis_matrix = basis_matrix[0:100,0:100].T
Vinv = np.linalg.inv(basis_matrix)
weights = Vinv[0,:]

# weights = np.linalg.solve(basis_matrix.T, np.ones(len(weights)).T)
norm = 2**(degree)*sp.factorial(degree)*np.sqrt(np.pi)

# integral = np.dot(pdf, weights*norm)

integral2 = np.dot(np.ones(len(weights)), weights)



# samples, weights = get_tensor_product_quadrature_rule(
#     degree+1,num_vars,gauss_hermite_pts_wts_1D)



from numpy.polynomial.hermite import hermfit, hermval
coef = hermfit(lejaPoints[:,0], lejaPoints[:,1], 2)



