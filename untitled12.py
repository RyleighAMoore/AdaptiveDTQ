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
        Px = Mesh[i,0]; Py = Mesh[i,1]
        if np.sqrt(Px**2 + Py**2) < 12*max(sigmaX,sigmaY):
            MeshToKeep.append([Px,Py])
            PdfToKeep.append(pdf[i])
            
    newMesh = np.asarray(MeshToKeep)
    plt.figure()
    plt.plot(Mesh[:,0], Mesh[:,1], '.k')
    plt.plot(newMesh[:,0], newMesh[:,1], '.r')
    plt.show()
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
    # poly.set_coefficients(coef)
    
    # indices = poly.indices
    # recursion_coeffs = np.asarray(poly.recursion_coeffs)
    
    # mesh = UM.generateOrderedGridCenteredAtZero(-0.5,0.5, -0.5,0.5, 0.1, includeOrigin=True)
    # # mesh = LP.generateLejaMeshNotCentered(200, min(sigmaX, sigmaX2), min(sigmaY,sigmaX2), 40, muX2, muY2)

    # vals1 =(poly.value((train_samples).T))
    
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(train_samples[:,0], train_samples[:,1], vals1, c='r', marker='.')
    # ax.scatter(train_samples[:,0], train_samples[:,1], train_values, c='k', marker='.')
    # print(np.max(train_values-vals1))
    print(coef[0])
    
    return coef[0], poly

    
import Functions as fun
h=0.01

degree = 35
muX = 0
muY = 0
sigmaX = 0.1
sigmaY = 0.1


muX2 = 0
muY2 = 0
sigmaX2=0.1
sigmaY2=0.1

train_samples = LP.generateLejaMeshNotCentered(300, sigmaX, sigmaY, 55, muX, muY)


# train_samples2 = UM.generateOrderedGridCenteredAtZero(-0.01,0.01, -0.01,0.01, 0.01, includeOrigin=False)
# train_samples = np.vstack((train_samples,train_samples2))
train_values = UM.generateICPDFShifted(train_samples[:,0], train_samples[:,1], sigmaX2, sigmaX2, muX2, muY2)
train_values = np.expand_dims(train_values,axis=1)

newPdf = []
sigmaX2s = []
sigmaY2s = []
muX2s = []
muY2s = []
Pdfs = []
Pdfs.append(np.asarray(train_values))
for i in range(3):
    print(i)
    newPdf = []
    for point in range(len(train_samples)):
        train_samples1 = np.copy(train_samples)
        Px = np.copy(train_samples[point,0])
        Py = np.copy(train_samples[point,1])
        muX2 = 2*Px - h*fun.f1(Px,Py); muX2s.append(sigmaX2)
        muY2 = 2*Py - h*fun.f2(Px,Py); muY2s.append(sigmaY2)
        sigmaX2 = np.sqrt(h)*fun.g1(); sigmaX2s.append(sigmaX2)
        sigmaY2 = np.sqrt(h)*fun.g2(); sigmaY2s.append(sigmaY2)
    
    
        mesh = UM.generateOrderedGridCenteredAtZero(-.2,.2, -.2,.2, 0.01, includeOrigin=True)
    
        dx1 = muX2*np.ones((1,len(train_samples))).T - Px*np.ones((1,len(train_samples))).T
        dy2 = muY2*np.ones((1,len(train_samples))).T - Py*np.ones((1,len(train_samples))).T
        delta2 = np.hstack((dx1,dy2))
        train_samples1 = train_samples1 - delta2
    
        integral, poly = QuadratureByInterpolation(train_samples1, train_values, sigmaX, sigmaY, muX, muY, degree)
        newPdf.append(integral)
    Pdfs.append(np.asarray(newPdf))
    train_values = np.asarray(newPdf)
fig = plt.figure()
ax = Axes3D(fig)
axes = plt.gca()
ax.scatter(train_samples[:,0], train_samples[:,1], Pdfs[0], c='r', marker='.')
ax.scatter(train_samples[:,0], train_samples[:,1], Pdfs[-1], c='k', marker='.')

