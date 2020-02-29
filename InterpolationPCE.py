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

def lejaIntegration(train_samples, train_values):
    # Configure polynomials
    degree=15
    univariate_variables = [norm(0,1),norm(0,1)]
    variable = IndependentMultivariateRandomVariable(univariate_variables)
    var_trans = AffineRandomVariableTransformation(variable)
    poly = PolynomialChaosExpansion()
    poly_opts = define_poly_options_from_variable_transformation(var_trans)
    poly.configure(poly_opts)
    indices = compute_hyperbolic_indices(poly.num_vars(),degree,1.0)
    # indices = compute_tensor_product_level_indices(poly.num_vars(),degree,max_norm=True)
    poly.set_indices(indices)
    
    num_vars = 2
    deriv_order= 0    
    probability_measure = True
    num_leja_samples = len(indices[0])-1
    initial_samples = np.asarray([[0],[0]])
    
    rv = multivariate_normal([0, 0], [[1, 0], [0, 1]])
    train_values = np.log(np.asarray([rv.pdf(train_samples)])).T
    
    
    basis_matrix = poly.basis_matrix(train_samples.T)
    assert np.size(basis_matrix,0) == np.size(basis_matrix,1)
    coef = np.linalg.lstsq(basis_matrix,train_values,rcond=None)[0]
    poly.set_coefficients(coef)

    
    e1vector = np.zeros((1,len(train_values)))
    e1vector[0,0] = 1
    testing = np.matmul(e1vector, np.linalg.inv(basis_matrix))
    aa = np.matmul(testing, np.ones(len(train_samples)))   
    aa2 = np.matmul(testing, np.exp(train_values))   
    aa3 = np.matmul(testing, train_values)   
    
    return aa, aa2
"""#Testing Interpolation
univariate_variables = [norm(0,1),norm(0,1)]
variable = IndependentMultivariateRandomVariable(univariate_variables)
var_trans = AffineRandomVariableTransformation(variable)
poly = PolynomialChaosExpansion()
poly_opts = define_poly_options_from_variable_transformation(var_trans)
poly.configure(poly_opts)
degree=15
indices = compute_hyperbolic_indices(poly.num_vars(),degree,1.0)
poly.set_indices(indices)
rv = multivariate_normal([0, 0], [[1, 0], [0, 1]])
num_vars = 2
deriv_order= 0    
probability_measure = True
num_leja_samples = len(indices[0])-1
initial_samples = np.asarray([[0],[0]])
train_samples = LP.generateLejaMesh(num_leja_samples, 1, 1, degree)  
train_values = np.log(np.asarray([rv.pdf(train_samples)])).T
aa, aa2 = lejaIntegration(train_samples, train_values) 
"""
   
def getLejaQuadratureRule(sigmaX, sigmaY):
    univariate_variables = [norm(0,sigmaX),norm(0,sigmaY)]
    variable = IndependentMultivariateRandomVariable(univariate_variables)
    var_trans = AffineRandomVariableTransformation(variable)
    poly = PolynomialChaosExpansion()
    poly_opts = define_poly_options_from_variable_transformation(var_trans)
    poly.configure(poly_opts)
    degree=20
    indices = compute_hyperbolic_indices(poly.num_vars(),degree,1.0)
    # indices = compute_tensor_product_level_indices(poly.num_vars(),degree,max_norm=True)
    poly.set_indices(indices)
    num_vars = 2
    deriv_order= 0    
    probability_measure = True
    num_leja_samples = len(indices[0])-1
    initial_samples = np.asarray([[0],[0]])
    train_samples = LP.generateLejaMesh(num_leja_samples, sigmaX, sigmaY, degree)
    
    rv = multivariate_normal([0, 0], [[sigmaX, 0], [0, sigmaY]])
    train_values = np.log(np.asarray([rv.pdf(train_samples)])).T

    basis_matrix = poly.basis_matrix(train_samples.T)
    assert np.size(basis_matrix,0) == np.size(basis_matrix,1)
    coef = np.linalg.lstsq(basis_matrix,train_values,rcond=None)[0]
    poly.set_coefficients(coef)
                
    e1vector = np.zeros((1,len(train_values)))
    e1vector[0,0] = 1
    weights = np.matmul(e1vector, np.linalg.inv(basis_matrix))
   
    return train_samples, weights

"""
train_samples, weights = getLejaQuadratureRule(1, 1)
aa = np.matmul(weights, np.ones(len(train_samples))) 
"""  


def LejaInterpolation(train_values,train_samples, sigmaX, sigmaY):
    univariate_variables = [norm(0,sigmaX),norm(0,sigmaY)]
    variable = IndependentMultivariateRandomVariable(univariate_variables)
    var_trans = AffineRandomVariableTransformation(variable)
    poly = PolynomialChaosExpansion()
    poly_opts = define_poly_options_from_variable_transformation(var_trans)
    poly.configure(poly_opts)
    
    degree=25
    indices = compute_hyperbolic_indices(poly.num_vars(),degree,1.0)
    # indices = compute_tensor_product_level_indices(poly.num_vars(),degree,max_norm=True)
    poly.set_indices(indices)
    
    num_vars = 2
    deriv_order= 0    
    probability_measure = True
    num_leja_samples = len(indices[0])-1
    initial_samples = np.asarray([[0],[0]])
    
    # train_samples, newLeja = LP.getLejaPoints(num_leja_samples, initial_samples,degree, num_candidate_samples = 5000, dimensions=num_vars)
    train_samples = LP.generateLejaMesh(num_leja_samples, sigmaX, sigmaY, degree)
    # train_samples = UM.generateRandomPoints(-4,4,-4,4,num_leja_samples+1)  # unordered mesh
    
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(train_samples[:,0], train_samples[:,1])
    # ax.scatter(train_samples[-1,0], train_samples[-1,1])
    
    # plt.show()
    
    
    rv = multivariate_normal([0, 0], [[sigmaX, 0], [0, sigmaX]])
    train_values = np.log(np.asarray([rv.pdf(train_samples)])).T
    
    
    basis_matrix = poly.basis_matrix(train_samples.T)
    # basis_matrix = basis_matrix[:,:np.size(basis_matrix,0)]
    # basis_matrix2 = poly.canonical_basis_matrix(train_samples.T)
    
    # precond_weights = christoffel_weights(basis_matrix)
    # precond_basis_matrix = precond_weights[:,np.newaxis]*basis_matrix
    # precond_train_values = precond_weights[:,np.newaxis]*train_values
    assert np.size(basis_matrix,0) == np.size(basis_matrix,1)
    coef = np.linalg.lstsq(basis_matrix,train_values,rcond=None)[0]
    poly.set_coefficients(coef)
    
    samples1 = np.asarray([[0],[0]])
    indices = poly.indices
    recursion_coeffs = np.asarray(poly.recursion_coeffs)
    
    # mesh, newLeja = LP.getLejaPoints(105, initial_samples, degree, num_candidate_samples = 5000, dimensions=num_vars)
    mesh = UM.generateOrderedGridCenteredAtZero(-4,4, -4,4, 0.1, includeOrigin=True)
    
    meshVals = np.asarray([rv.pdf(mesh)]).T
    
    vals1 = np.exp(poly.value(mesh.T))
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(train_samples[:,0], train_samples[:,1], np.exp(train_values), c='k', s=15)
    ax.scatter(mesh[:,0], mesh[:,1], vals1, c='r', s=10)
    
    
    e1vector = np.zeros((1,len(train_values)))
    e1vector[0,0] = 1
    testing = np.matmul(e1vector, np.linalg.inv(basis_matrix))
    aa = np.matmul(testing, np.ones(len(train_samples)))   
    aa2 = np.matmul(testing, np.exp(train_values))   
    aa3 = np.matmul(testing, train_values)  
    
    
    

    

# univariate_variables = [norm(0,1),norm(0,1)]
# variable = IndependentMultivariateRandomVariable(univariate_variables)

# var_trans = AffineRandomVariableTransformation(variable)
# poly = PolynomialChaosExpansion()
# poly_opts = define_poly_options_from_variable_transformation(var_trans)
# poly.configure(poly_opts)

# degree=15
# indices = compute_hyperbolic_indices(poly.num_vars(),degree,1.0)
# # indices = compute_tensor_product_level_indices(poly.num_vars(),degree,max_norm=True)
# poly.set_indices(indices)

# num_vars = 2
# deriv_order= 0    
# probability_measure = True
# num_leja_samples = len(indices[0])-1
# initial_samples = np.asarray([[0],[0]])

# # train_samples, newLeja = LP.getLejaPoints(num_leja_samples, initial_samples,degree, num_candidate_samples = 5000, dimensions=num_vars)
# # train_samples = LP.generateLejaMesh(num_leja_samples, 1, 1, degree)
# train_samples = UM.generateRandomPoints(-4,4,-4,4,num_leja_samples+1)  # unordered mesh

# # fig = plt.figure()
# # ax = Axes3D(fig)
# # ax.scatter(train_samples[:,0], train_samples[:,1])
# # ax.scatter(train_samples[-1,0], train_samples[-1,1])

# # plt.show()


# rv = multivariate_normal([0, 0], [[1, 0], [0, 1]])
# train_values = np.log(np.asarray([rv.pdf(train_samples)])).T


# basis_matrix = poly.basis_matrix(train_samples.T)
# # basis_matrix = basis_matrix[:,:np.size(basis_matrix,0)]
# # basis_matrix2 = poly.canonical_basis_matrix(train_samples.T)

# precond_weights = christoffel_weights(basis_matrix)
# precond_basis_matrix = precond_weights[:,np.newaxis]*basis_matrix
# # precond_train_values = precond_weights[:,np.newaxis]*train_values
# assert np.size(basis_matrix,0) == np.size(basis_matrix,1)
# coef = np.linalg.lstsq(basis_matrix,train_values,rcond=None)[0]
# poly.set_coefficients(coef)

# samples1 = np.asarray([[0],[0]])
# indices = poly.indices
# recursion_coeffs = np.asarray(poly.recursion_coeffs)

# mesh, newLeja = LP.getLejaPoints(105, initial_samples, degree, num_candidate_samples = 5000, dimensions=num_vars)
# # mesh = UM.generateOrderedGridCenteredAtZero(-3,3, -3,3, 0.4, includeOrigin=True)

# meshVals = np.asarray([rv.pdf(mesh)]).T

# vals1 = np.exp(poly.value(mesh.T))

# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(train_samples[:,0], train_samples[:,1], np.exp(train_values), c='k', s=15)
# ax.scatter(mesh[:,0], mesh[:,1], vals1, c='r', s=10)


# e1vector = np.zeros((1,len(train_values)))
# e1vector[0,0] = 1
# testing = np.matmul(e1vector, np.linalg.inv(basis_matrix))
# aa = np.matmul(testing, np.ones(len(train_samples)))   
# aa2 = np.matmul(testing, np.exp(train_values))   
# aa3 = np.matmul(testing, train_values)   



import Functions as fun
from scipy.interpolate import griddata
import UnorderedMesh as UM
from tqdm import tqdm, trange

def stepForwardInTime(Mesh, Pdf, h):
    PdfNew = []
    train_samples = Mesh
    train_values = Pdf
    lejaPoints, weights = getLejaQuadratureRule(np.sqrt(h*fun.g1()), np.sqrt(h*fun.g2()))
    # plt.scatter(lejaPoints[:,0], lejaPoints[:,1])
    for point in trange(len(train_samples)):
        Px = train_samples[point,0]
        Py = train_samples[point,1]
        dx = Px*np.ones((1,len(train_samples))).T
        dy = Py*np.ones((1,len(train_samples))).T
        delta = np.hstack((dx,dy))
        train_samples = train_samples - delta
        
        Px = train_samples[point,0]
        Py = train_samples[point,1]
        grid_z2 = griddata(train_samples, train_values, lejaPoints, method='cubic', fill_value=0)
        aa = np.dot(weights, grid_z2) 
        PdfNew.append(np.copy(aa))
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # ax.scatter(lejaPoints[:,0], lejaPoints[:,1], grid_z2, c='r', marker='.')
        # print(aa)
    return np.squeeze(PdfNew)

# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(train_samples[:,0], train_samples[:,1], PdfNew, c='r', marker='.')
# ax.scatter(train_samples[:,0], train_samples[:,1], train_value[1], c='k', marker='.')

"""
import pickle
pickle_in = open("C:/Users/Rylei/Documents/SimpleDTQ/PickledData/PdfTraj.p","rb")
train_value = pickle.load(pickle_in)
train_values = np.asarray([train_value[0]]).T


pickle_in = open("C:/Users/Rylei/Documents/SimpleDTQ/PickledData/Mesh.p","rb")
train_samples = pickle.load(pickle_in)


PdfNew = stepForwardInTime(train_samples,train_values, 0.01)
"""





# import Functions as fun
# from scipy.interpolate import griddata
# import UnorderedMesh as UM
# # def stepForwardInTime(Mesh, Pdf, h):
# PdfNew = []
# # train_samples = Mesh
# # train_values = Pdf
# import pickle
# pickle_in = open("C:/Users/Rylei/Documents/SimpleDTQ/PickledData/PdfTraj.p","rb")
# train_value = pickle.load(pickle_in)
# train_values = train_value[0]


# pickle_in = open("C:/Users/Rylei/Documents/SimpleDTQ/PickledData/Mesh.p","rb")
# train_sample = pickle.load(pickle_in)
# train_samples = train_sample

# from tqdm import tqdm, trange

# h=0.01
# lejaPoints, weights = getLejaQuadratureRule(np.sqrt(h*fun.g1()), np.sqrt(h*fun.g2()))
# # plt.scatter(lejaPoints[:,0], lejaPoints[:,1])
# for point in trange(len(train_samples)):
#     Px = train_samples[point,0]; Py = train_samples[point,1]
#     # grid_x, grid_y = np.mgrid[-6:6:100j, -6:6:100j]
#     dx = Px*np.ones((1,len(train_samples))).T
#     dy = Py*np.ones((1,len(train_samples))).T
#     delta = np.hstack((dx,dy))
#     train_samples = train_samples - delta
#     Px = train_samples[point,0]; Py = train_samples[point,1]

#     pointVals = []
#     gridz2 = []
#     # for j in range(len(lejaPoints)):
#     #     pdfvals = []
#     #     nearest3, distToNearestPoints, indices = UM.findNearestKPoints(lejaPoints[j,0],lejaPoints[j,1], train_samples, 3, getIndices=True)
#     #     pdfvals.append(train_values[indices[0]]); pdfvals.append(train_values[indices[1]]); pdfvals.append(train_values[indices[2]])
#     #     value = UM.baryInterp(Px, Py, nearest3, np.asarray(pdfvals), nearestNeighbor=True)
#     #     gridz2.append(np.copy(value))
#     #     # print(value)
#     # grid_z2 = np.asarray(gridz2)
        
#     grid_z2 = griddata(train_samples, train_values, lejaPoints, method='cubic', fill_value=0)
#     aa = np.dot(weights, grid_z2) 
#     PdfNew.append(np.copy(aa))
#     # fig = plt.figure()
#     # ax = Axes3D(fig)
#     # ax.scatter(lejaPoints[:,0], lejaPoints[:,1], grid_z2, c='r', marker='.')
#     # print(aa)
#     # return PdfNew

# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(train_samples[:,0], train_samples[:,1], PdfNew, c='r', marker='.')
# ax.scatter(train_samples[:,0], train_samples[:,1], train_value[1], c='k', marker='.')



# PdfNew = stepForwardInTime(mesh, surfaces[0], h)
# PdfNew = []
# train_samples = meshSoln
# train_values = pdfSoln[0]
# lejaPoints, weights = getLejaQuadratureRule(np.sqrt(h*fun.g1()), np.sqrt(h*fun.g2()))
# for point in range(10):
#     Px = train_samples[point,0]; Py = train_samples[point,1]
#     # grid_x, grid_y = np.mgrid[-6:6:100j, -6:6:100j]
#     dx = Px*np.ones((1,len(train_samples))).T
#     dy = Py*np.ones((1,len(train_samples))).T
#     delta = np.hstack((dx,dy))
#     train_samples = train_samples + delta
    
#     grid_z2 = griddata(train_samples, train_values, lejaPoints, method='cubic', fill_value=0)
#     aa = np.dot(weights, grid_z2) 
#     PdfNew.append(aa)





# train_samples = meshSoln
# train_values = pdfSoln[1]
# Px = train_samples[4,0]
# Py = train_samples[4,1]

# grid_x, grid_y = np.mgrid[-6:6:100j, -6:6:100j]
# train_samples1, weights = getLejaQuadratureRule(.1, .1)
# dx = Px*np.ones((1,len(train_samples1))).T
# dy = Py*np.ones((1,len(train_samples1))).T
# delta = np.hstack((dx,dy))
# train_samples1 = train_samples1 + delta


# from scipy.interpolate import griddata

# grid_z2 = griddata(train_samples, train_values, train_samples1, method='cubic', fill_value=0)
# grid_z = griddata(train_samples, train_values, (grid_x,grid_y), method='cubic', fill_value=0)

# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(train_samples1[:,0],train_samples1[:,1], grid_z2, c='r', marker='*')

# ax.scatter(grid_x,grid_y, grid_z, c='k', marker='.')


# aa = np.dot(weights, grid_z2) 




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


