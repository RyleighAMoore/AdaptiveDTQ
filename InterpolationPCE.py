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
from pyapprox.indexing import compute_hyperbolic_indices, tensor_product_indices
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

univariate_variables = [norm(0,1),norm(0,1)]
variable = IndependentMultivariateRandomVariable(univariate_variables)

var_trans = AffineRandomVariableTransformation(variable)
poly = PolynomialChaosExpansion()
poly_opts = define_poly_options_from_variable_transformation(var_trans)
poly.configure(poly_opts)

degree=15
indices = compute_hyperbolic_indices(poly.num_vars(),degree,1.0)
poly.set_indices(indices)

num_vars = 2
deriv_order=0    
probability_measure = True
num_leja_samples = 100
initial_samples = np.asarray([[0],[0]])

train_samples, newLeja = LP.getLejaPoints(num_leja_samples, initial_samples,degree, num_candidate_samples = 5000, dimensions=num_vars)
rv = multivariate_normal([0, 0], [[1, 0], [0, 1]])
train_values = np.asarray([rv.pdf(train_samples)]).T


basis_matrix = poly.basis_matrix(train_samples.T)
precond_weights = christoffel_weights(basis_matrix)
precond_basis_matrix = precond_weights[:,np.newaxis]*basis_matrix
precond_train_values = precond_weights[:,np.newaxis]*train_values
coef = np.linalg.lstsq(precond_basis_matrix,precond_train_values,rcond=None)[0]
poly.set_coefficients(coef)

samples1 = np.asarray([[0],[0]])
mesh = UM.generateOrderedGridCenteredAtZero(-5, 5, -5, 5, 0.1)
indices = poly.indices
recursion_coeffs = np.asarray(poly.recursion_coeffs)

vals1 = poly.value(mesh.T)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(mesh[:,0], mesh[:,1], vals1, c='r', marker='.')
ax.scatter(train_samples[:,0], train_samples[:,1], train_values, '*k')


grid_x, grid_y = np.mgrid[-2:2:100j, -2:2:200j]
from scipy.interpolate import griddata

grid_z2 = griddata(train_samples, train_values, (grid_x, grid_y), method='cubic')
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(grid_x,grid_y, grid_z2, c='r', marker='.')

# vals = evaluate_multivariate_orthonormal_polynomial(
#         samples1, indices, recursion_coeffs,deriv_order=0,
#         basis_type_index_map=None)

# def test_evaluate_multivariate_orthonormal_polynomial(self):
#         num_vars = 2; alpha = 0.; beta = 0.; degree = 2; deriv_order=1    
#         probability_measure = True

#         ab = jacobi_recurrence(
#             degree+1,alpha=alpha,beta=beta,probability=probability_measure)

#         x,w=np.polynomial.legendre.leggauss(degree)
#         samples = cartesian_product([x]*num_vars,1)
#         weights = outer_product([w]*num_vars)

#         indices = compute_hyperbolic_indices(num_vars,degree,1.0)

#         # sort lexographically to make testing easier
#         I = np.lexsort((indices[0,:],indices[1,:], indices.sum(axis=0)))
#         indices = indices[:,I]

#         basis_matrix = evaluate_multivariate_orthonormal_polynomial(
#             samples,indices,ab,deriv_order)

#         exact_basis_vals_1d = []
#         exact_basis_derivs_1d = []
#         for dd in range(num_vars):
#             x = samples[dd,:]
#             exact_basis_vals_1d.append(
#                 np.asarray([1+0.*x,x,0.5*(3.*x**2-1)]).T)
#             exact_basis_derivs_1d.append(np.asarray([0.*x,1.0+0.*x,3.*x]).T)
#             exact_basis_vals_1d[-1]/=np.sqrt(1./(2*np.arange(degree+1)+1))
#             exact_basis_derivs_1d[-1]/=np.sqrt(1./(2*np.arange(degree+1)+1))

#         exact_basis_matrix = np.asarray(
#             [exact_basis_vals_1d[0][:,0],exact_basis_vals_1d[0][:,1],
#                  exact_basis_vals_1d[1][:,1],exact_basis_vals_1d[0][:,2],
#             exact_basis_vals_1d[0][:,1]*exact_basis_vals_1d[1][:,1],
#             exact_basis_vals_1d[1][:,2]]).T

#         # x1 derivative
#         exact_basis_matrix = np.vstack((exact_basis_matrix,np.asarray(
#             [0.*x,exact_basis_derivs_1d[0][:,1],0.*x,
#              exact_basis_derivs_1d[0][:,2],
#              exact_basis_derivs_1d[0][:,1]*exact_basis_vals_1d[1][:,1],
#              0.*x]).T))

#         # x2 derivative
#         exact_basis_matrix = np.vstack((exact_basis_matrix,np.asarray(
#             [0.*x,0.*x,exact_basis_derivs_1d[1][:,1],0.*x,
#             exact_basis_vals_1d[0][:,1]*exact_basis_derivs_1d[1][:,1],
#             exact_basis_derivs_1d[1][:,2]]).T))

#         func = lambda x: evaluate_multivariate_orthonormal_polynomial(
#             x,indices,ab,0)
#         basis_matrix_derivs = basis_matrix[samples.shape[1]:]
#         basis_matrix_derivs_fd = np.empty_like(basis_matrix_derivs)
#         for ii in range(samples.shape[1]):
#             basis_matrix_derivs_fd[ii::samples.shape[1],:] = approx_fprime(
#                 samples[:,ii:ii+1],func,1e-7)
#         assert np.allclose(
#             exact_basis_matrix[samples.shape[1]:], basis_matrix_derivs_fd)

#         assert np.allclose(exact_basis_matrix, basis_matrix)


