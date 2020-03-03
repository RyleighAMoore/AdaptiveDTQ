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

# import dill
# dill.load_session('session.pkl')

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
   
def getLejaQuadratureRule(sigmaX, sigmaY, muX, muY):
    univariate_variables = [norm(muX,sigmaX),norm(muY,sigmaY)]
    variable = IndependentMultivariateRandomVariable(univariate_variables)
    var_trans = AffineRandomVariableTransformation(variable)
    
    poly = PolynomialChaosExpansion()
    poly_opts = define_poly_options_from_variable_transformation(var_trans)
    poly.configure(poly_opts)
    degree=7
    indices = compute_hyperbolic_indices(poly.num_vars(),degree,1.0)
    # indices = compute_tensor_product_level_indices(poly.num_vars(),degree,max_norm=True)
    poly.set_indices(indices)
    num_vars = 2
    deriv_order= 0    
    probability_measure = True
    
    num_leja_samples = len(indices[0])-1
    initial_samples = np.asarray([[muX],[muY]])
    train_samples = LP.generateLejaMeshNotCentered(num_leja_samples, sigmaX, sigmaY, degree,muX,muY)
    
    # rv = multivariate_normal([muX, muY], [[sigmaX**2, 0], [0, sigmaY**2]])
    # train_values = np.asarray([rv.pdf(train_samples)]).T

    basis_matrix = poly.basis_matrix(train_samples.T)
    precond_weights = christoffel_weights(basis_matrix)
    precond_basis_matrix = precond_weights[:,np.newaxis]*basis_matrix
    # precond_train_values = precond_weights[:,np.newaxis]*train_values
    assert np.size(basis_matrix,0) == np.size(basis_matrix,1)
    # coef = np.linalg.lstsq(precond_basis_matrix,precond_train_values,rcond=None)[0]
    # poly.set_coefficients(coef)
                
    e1vector = np.zeros((1,len(train_samples)))
    e1vector[0,0] = 1
    weights = np.matmul(e1vector, np.linalg.inv(basis_matrix))
   
    return train_samples, weights, poly

'''
# Quadrature Testing
# train_samples1, weights1, poly = getLejaQuadratureRule(0.1, 0.1 ,1,1)
# train_samples, weights, poly = getLejaQuadratureRule(0.1, 0.1,0,0)
# plt.scatter(train_samples1[:,0], train_samples1[:,1])

# # train_samples1, weights, poly = getLejaQuadratureRule(1, 1, 0,0)
# Aa1 = np.matmul(weights1, np.ones(len(train_samples1))) # should be 1
# Aa = np.matmul(weights, np.ones(len(train_samples))) 
# rv = multivariate_normal([0, 0], [[0.25, 0], [0, 0.25]])
# vals1 = np.asarray([rv.pdf(train_samples1)]).T  
# vals = np.asarray([rv.pdf(train_samples)]).T
# Ab1 = np.matmul(weights1, vals1) # should be 2.277E-44
# Ab = np.matmul(weights, vals) # Should be 0.612134

# rv = multivariate_normal([1, -1], [[0.25, 0], [0, 0.25]])
# vals1 = np.asarray([rv.pdf(train_samples1)]).T 
# vals = np.asarray([rv.pdf(train_samples)]).T
# Ac1 = np.matmul(weights1, vals1) # should be 0.000279332
# Ac = np.matmul(weights, vals) # Should be 0.0130763
'''


def LejaInterpolation(train_values,train_samples, sigmaX, sigmaY):
    univariate_variables = [norm(0,sigmaX),norm(0,sigmaY)]
    variable = IndependentMultivariateRandomVariable(univariate_variables)
    var_trans = AffineRandomVariableTransformation(variable)
    poly = PolynomialChaosExpansion()
    poly_opts = define_poly_options_from_variable_transformation(var_trans)
    poly.configure(poly_opts)
    
    degree=9
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
    
    


def interpolateLeja(Mesh, Pdf, newPoint, h):
    train_samples = Mesh
    train_values = np.asarray([Pdf]).T
    # lejaPoints, weights = getLejaQuadratureRule(np.sqrt(h*fun.g1()), np.sqrt(h*fun.g2()))
    # plt.scatter(lejaPoints[:,0], lejaPoints[:,1])
    # Px = train_samples[point,0]
    # Py = train_samples[point,1]
    # dx = Px*np.ones((1,len(train_samples))).T
    # dy = Py*np.ones((1,len(train_samples))).T
    # delta = np.hstack((dx,dy))
    # train_samples = train_samples - delta
    # Px = train_samples[point,0]
    # Py = train_samples[point,1]
    grid_z2 = griddata(train_samples, train_values, newPoint, method='cubic', fill_value=0)
        
    return grid_z2

# interp2 = interpolateLeja(Meshes[0], pdf, np.asarray([[2],[1]]).T, 0.1)



import Functions as fun
from scipy.interpolate import griddata
import UnorderedMesh as UM
from tqdm import tqdm, trange
h=0.01
lejaPoints1, weights, poly = getLejaQuadratureRule(np.sqrt(h)*fun.g1(), np.sqrt(h)*fun.g2(), 0,0)
# lejaPoints1, weights = np.polynomial.hermite.hermgauss(10)
def stepForwardInTime(Mesh, Pdf, h):
    print("Stepping Forward...")
    PdfNew = []
    train_samples = np.copy(Mesh)
    train_values = np.copy(Pdf)
    # If g1 or g2 changes spatially, we need to move this inside the loop.
    # plt.scatter(lejaPoints[:,0], lejaPoints[:,1])
    # lejaPoints1, weights, poly = getLejaQuadratureRule(np.sqrt(h)*fun.g1(), np.sqrt(h)*fun.g2(), 0,0)
    for point in range(len(train_samples)):
        train_samples = np.copy(Mesh)
        lejaPoints = np.copy(lejaPoints1)
        Px = np.copy(train_samples[point,0])
        Py = np.copy(train_samples[point,1])
        muX = Px - h*fun.f1(Px,Py)
        muY = Py - h*fun.f2(Px,Py)
        # lejaPoints, weights, poly = getLejaQuadratureRule(np.sqrt(h)*fun.g1(), np.sqrt(h)*fun.g2(), muX,muY)

        dx1 = muX*np.ones((1,len(lejaPoints))).T
        dy2 = muY*np.ones((1,len(lejaPoints))).T
        delta2 = np.hstack((dx1,dy2))
        lejaPoints = lejaPoints + delta2
    
       
        # dx = Px*np.ones((1,len(train_samples))).T #+ h*fun.f1(Px,Py)*np.ones((1,len(train_samples))).T
        # dy = Py*np.ones((1,len(train_samples))).T #+ h*fun.f2(Px,Py)*np.ones((1,len(train_samples))).T
        # delta = np.hstack((dx,dy))
        # train_samples = train_samples - delta
        # print(train_samples[0,0])
        # print(train_samples[0,1])
        # mesh = UM.generateOrderedGridCenteredAtZero(-4,4, -4,4, 0.1, includeOrigin=True)
        # rv = multivariate_normal([muX, muY], [[np.sqrt(h)*fun.g1(),0], [0, np.sqrt(h)*fun.g2()]])
        # normal = (np.asarray([rv.pdf(mesh)])).T
        mesh = UM.generateOrderedGridCenteredAtZero(-2,2, -2,2, 0.01, includeOrigin=True)
        # rv = multivariate_normal([muX, muY], [[np.sqrt(h)*fun.g1(),0], [0, np.sqrt(h)*fun.g2()]])
        
        # rv1 = multivariate_normal([0, 0], [[0.25,0], [0, 0.25]])
        # normal1 = (np.asarray([rv.pdf(mesh)])).T
        
    
        grid_z2 = griddata(train_samples, train_values, lejaPoints, method='cubic', fill_value=0)
        grid_z = griddata(train_samples, train_values, mesh, method='cubic', fill_value=0)
        # grid_z3 = (np.asarray([rv1.pdf(lejaPoints1)])).T
        
        
        
        aa = np.dot(weights, grid_z2) 
        PdfNew.append(np.copy(aa))
        
        # mesh = UM.generateOrderedGridCenteredAtZero(-4,4, -4,4, 0.1, includeOrigin=True)    
        # vals1 = np.exp(poly.value(mesh.T))
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # ax.scatter(mesh[:,0], mesh[:,1], vals1, c='r', marker='.')
        # ax.scatter(train_samples[:,0], train_samples[:,1], train_values, c='g', marker='.')
        
        
        # print(aa)
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # ax.scatter(lejaPoints[:,0], lejaPoints[:,1], grid_z2, c='r', marker='*')
        # ax.scatter(train_samples[:,0], train_samples[:,1], train_values, c='g', marker='.')
        # ax.scatter(train_samples[0,0], train_samples[0,1], train_values, c='g', marker='*')
        # ax.scatter(mesh[:,0], mesh[:,1], normal1, c='k', marker='.')
        # ax.scatter(lejaPoints[:,0], lejaPoints[:,1], grid_z3, c='k', marker='.')
        # ax.scatter(mesh[:,0], mesh[:,1], grid_z, c='k', marker='.')

        # ax.scatter(mesh[:,0], mesh[:,1], vals1, c='r', marker='.')


        # fig = plt.figure()
        # ax = Axes3D(fig)
        # ax.scatter(Mesh[0,0], Mesh[0,1], Pdf, c='b', marker='*')
        # ax.scatter(Mesh[:,0], Mesh[:,1], Pdf, c='b', marker='*')
        # ax.scatter(mesh[:,0], mesh[:,1], normal, c='k', marker='.')

        

        
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # ax.scatter(Mesh[0,0], Mesh[0,1], Pdf, c='y', marker='*')
        # ax.scatter(Mesh[:,0], Mesh[:,1], Pdf, c='b', marker='*')

        

        # print(aa)
        
    mesh = UM.generateOrderedGridCenteredAtZero(-2,2, -2,2, 0.01, includeOrigin=True)
    grid_z = griddata(train_samples, train_values, mesh, method='cubic', fill_value=0)
    integral = 0.01**2*np.sum(grid_z)
    print(integral)

    PdfNew = np.asarray(PdfNew)
    return np.squeeze(PdfNew)

# import pickle
# pickle_in = open("C:/Users/Rylei/Documents/SimpleDTQ/PickledData/PdfTraj1.p","rb")
# train_value = pickle.load(pickle_in)
# train_values = np.asarray([train_value[0]]).T

# pickle_in = open("C:/Users/Rylei/Documents/SimpleDTQ/PickledData/Mesh1.p","rb")
# train_samples = pickle.load(pickle_in)

# PdfNew = stepForwardInTime(train_samples,train_values, 0.01)
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(train_samples[:,0], train_samples[:,1], PdfNew, c='r', marker='.')
# ax.scatter(train_samples[:,0], train_samples[:,1], train_value[31], c='k', marker='.')

# t=0


