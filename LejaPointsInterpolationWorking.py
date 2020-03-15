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

def getMeshValsThatAreClose(Mesh, pdf, sigmaX, sigmaY, muX, muY):
    MeshToKeep = []
    PdfToKeep = []
    for i in range(len(Mesh)):
        Px = Mesh[i,0] - muX; Py = Mesh[i,1] - muY
        if np.sqrt(Px**2 + Py**2) < 8*max(sigmaX,sigmaY):
            MeshToKeep.append([Px,Py])
            PdfToKeep.append(pdf[i])
    return np.asarray(MeshToKeep), np.asarray(PdfToKeep)



def QuadratureByInterpolation(train_samples, train_values, sigmaX, sigmaY, muX, muY, degree):
    # train_samples, train_values = getMeshValsThatAreClose(train_samples, train_values, sigmaX, sigmaY, muX, muY)
    
    univariate_variables = [norm(muX,sigmaX),norm(muY,sigmaY)]
    variable = IndependentMultivariateRandomVariable(univariate_variables)
    var_trans = AffineRandomVariableTransformation(variable)
    
    poly = PolynomialChaosExpansion()
    poly_opts = define_poly_options_from_variable_transformation(var_trans)
    poly.configure(poly_opts)
    degree=degree
    # indices = compute_hyperbolic_indices(poly.num_vars(),degree,1.0)
    # indices = compute_tensor_product_level_indices(poly.num_vars(),degree,max_norm=True)
    degrees = [int(train_samples.shape[0]**(1/poly.num_vars()))]*poly.num_vars()
    indices = tensor_product_indices(degrees)
    poly.set_indices(indices)

    
    basis_matrix = poly.basis_matrix(train_samples.T)
    numRows = np.size(basis_matrix,0)
    numCols = np.size(basis_matrix,1)
    ZeroCoeffs = np.zeros((numCols-numRows,1))
    assert len(ZeroCoeffs) >= 0
    
    basis_matrix = basis_matrix[:,:numRows]
    
    precond_weights = christoffel_weights(basis_matrix)
    precond_basis_matrix = precond_weights[:,np.newaxis]*basis_matrix
    precond_train_values = precond_weights[:,np.newaxis]*train_values
    
    assert np.size(basis_matrix,0) == np.size(basis_matrix,1)
    coef = np.linalg.lstsq(precond_basis_matrix,precond_train_values,rcond=None)[0]
    coef = np.vstack((coef, ZeroCoeffs))
    poly.set_coefficients(coef)
    
    indices = poly.indices
    recursion_coeffs = np.asarray(poly.recursion_coeffs)
    
    mesh = UM.generateOrderedGridCenteredAtZero(-0.5,0.5, -0.5,0.5, 0.1, includeOrigin=True)
    # mesh = LP.generateLejaMeshNotCentered(200, min(sigmaX, sigmaX2), min(sigmaY,sigmaX2), 40, muX2, muY2)

    vals1 =(poly.value((train_samples).T))
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(train_samples[:,0], train_samples[:,1], vals1, c='r', marker='.')
    ax.scatter(train_samples[:,0], train_samples[:,1], train_values, c='k', marker='.')
    print(np.max(train_values-vals1))
    print(coef[0])
    
    return coef[0], poly
    
    

degree = 35
muX = -0.04
muY = 0
sigmaX = 0.05
sigmaY = 0.05

muX2 = 0
muY2 = 0
sigmaX2=0.2
sigmaY2=0.2

train_samples = LP.generateLejaMeshNotCentered(300, sigmaX, sigmaY, 35, muX, muY)
# train_samples2 = LP.generateLejaMeshNotCentered(100, sigmaX2, sigmaY2, 25, muX2, muY2)
# train_samples = UM.generateOrderedGridCenteredAtZero(-4,4, -4,4, 0.1, includeOrigin=True)
# train_samples = np.vstack((train_samples,train_samples2))
train_values = UM.generateICPDFShifted(train_samples[:,0], train_samples[:,1], sigmaX2, sigmaX2, muX2, muY2)
# for i in range(len(train_samples)):
#     r = train_samples[i,0]**2+train_samples[i,1]**2
#     # train_values[i] = train_samples[i,0]**3
#     train_values[i] = np.sin(train_samples[i,0]+ train_samples[i,1])**2



rv1 = multivariate_normal([muX, muY], [[sigmaX**2,0], [0, sigmaY**2]])
# mesh = UM.generateOrderedGridCenteredAtZero(-.2,.2, -.2,.2, 0.01, includeOrigin=True)
mesh = LP.generateLejaMeshNotCentered(300, sigmaX, sigmaY, 40, muX, muY)
normal1 = (np.asarray([rv1.pdf(mesh)])).T
import Functions as fun
point = 0
h=0.01
Px = np.copy(train_samples[point,0])
Py = np.copy(train_samples[point,1])
# muX = Px - h*fun.f1(Px,Py)
# muY = Py - h*fun.f2(Px,Py)
# sigmaX = np.sqrt(h)*fun.g1()
# sigmaY = np.sqrt(h)*fun.g2()
# degree = 12

dx1 = muX2*np.ones((1,len(train_samples))).T + Px*np.ones((1,len(train_samples))).T
dy2 = muY2*np.ones((1,len(train_samples))).T + Py*np.ones((1,len(train_samples))).T
delta2 = np.hstack((dx1,dy2))
# train_samples = train_samples - delta2

train_values = np.expand_dims(train_values,axis=1)
integral, poly = QuadratureByInterpolation(train_samples, train_values, sigmaX, sigmaY, muX, muY, degree)


fig = plt.figure()
ax = Axes3D(fig)
# axes = plt.gca()
# axes.set_ylim([-1,1])
# axes.set_ylim([-1,1])
# axes.set_zlim([-1,10])

ax.scatter(train_samples[:,0], train_samples[:,1], train_values, c='r', marker='.')
ax.scatter(mesh[:,0], mesh[:,1], normal1, c='k', marker='.')
# ax.scatter(mesh[:,0], mesh[:,1], normal1, c='k', marker='.')

