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

   
def getLejaQuadratureRule(sigmaX, sigmaY, muX, muY):
    univariate_variables = [norm(muX,sigmaX),norm(muY,sigmaY)]
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
    initial_samples = np.asarray([[muX],[muY]])
    train_samples = LP.generateLejaMeshNotCentered(num_leja_samples, sigmaX, sigmaY, degree,muX,muY)


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
   
    return train_samples, weights


def newIntegrand(x1,x2,mesh,h):
    y1 = mesh[:,0]
    y2 = mesh[:,1]
    scale = h*g1(x1,x2)*g2(x1,x2)/(h*g1(y1,y2)*g2(y1,y2))
    val = scale*np.exp(-(h**2*f1(y1,y2)**2+2*h*f1(y1,y2)*(x1-y1))/(2*h*g1(x1,x2)**2) + -(h**2*f2(y1,y2)**2+2*h*f2(y1,y2)*(x2-y2))/(2*h*g2(x1,x2)**2))*np.exp((x1-y1+h*f1(y1,y2))**2/(2*h*g1(x1,x2)**2) - (x1-y1+h *f1(y1,y2))**2/(2*h*g1(y1,y2)**2) + (x2-y2+h*f2(y1,y2))**2/(2*h*g2(x1,x2)**2) - (x2-y2+h* f2(y1,y2))**2/(2*h*g2(y1,y2)**2))
    val = scale*np.exp(-(h**2*f1(y1,y2)**2-2*h*f1(y1,y2)*(x1-y1))/(2*h*g1(x1,x2)**2) + -(h**2*f2(y1,y2)**2-2*h*f2(y1,y2)*(x2-y2))/(2*h*g2(x1,x2)**2))*np.exp((x1-y1-h*f1(y1,y2))**2/(2*h*g1(x1,x2)**2) - (x1-y1-h *f1(y1,y2))**2/(2*h*g1(y1,y2)**2) + (x2-y2-h*f2(y1,y2))**2/(2*h*g2(x1,x2)**2) - (x2-y2-h* f2(y1,y2))**2/(2*h*g2(y1,y2)**2))
    return val


def getMeshValsThatAreClose(Mesh, pdf, sigmaX, sigmaY, muX, muY):
    MeshToKeep = []
    PdfToKeep = []
    distances = np.sqrt((Mesh[:,0]-muX)**2 + (Mesh[:,1]-muY)**2)
    
    for i in range(len(Mesh)):
        Px = Mesh[i,0]; Py = Mesh[i,1]
        if np.sqrt((Px-muX)**2 + (Py-muY)**2) < 5*max(sigmaX,sigmaY):
            MeshToKeep.append([Px,Py])
            PdfToKeep.append(pdf[i])
    # if len(Mesh)> len(MeshToKeep):        
    #     fig = plt.figure()
    #     ax = Axes3D(fig)
    #     ax.scatter(np.asarray(MeshToKeep)[:,0], np.asarray(MeshToKeep)[:,1], np.asarray(PdfToKeep), c='r', marker='.')
    #     # ax.scatter(Mesh[:,0], Mesh[:,1], pdf, c='k', marker='.')
    #     ax.scatter(muX, muY, 0, c='g', marker='*')
    return np.asarray(MeshToKeep), np.asarray(PdfToKeep)


def QuadratureByInterpolation(train_samples, train_values, sigmaX, sigmaY, muX, muY, degree):
    
    univariate_variables = [norm(muX,sigmaX),norm(muY,sigmaY)]
    variable = IndependentMultivariateRandomVariable(univariate_variables)
    var_trans = AffineRandomVariableTransformation(variable)
    
    poly = PolynomialChaosExpansion()
    poly_opts = define_poly_options_from_variable_transformation(var_trans)
    poly.configure(poly_opts)
    degree=degree
    num_vars = 2
    indices = compute_hyperbolic_indices(poly.num_vars(),degree,1.0)
    # indices = compute_tensor_product_level_indices(poly.num_vars(),degree,max_norm=True)
    # degrees = [int(train_samples.shape[0]**(1/poly.num_vars()))]*poly.num_vars()
    # indices = tensor_product_indices(degrees)
    poly.set_indices(indices)

    # train_values = np.log(train_values)
    basis_matrix = poly.basis_matrix(train_samples.T)
    numRows = np.size(basis_matrix,0)
    numCols = np.size(basis_matrix,1)
    ZeroCoeffs = np.zeros((numCols-numRows,1))
    assert len(ZeroCoeffs) >= 0
    
    basis_matrix = basis_matrix[:,:numRows]
    
    Vinv = np.linalg.inv(basis_matrix)
    print("cond=",np.sum(np.abs(Vinv[0,:])))
    
    # precond_weights = christoffel_weights(basis_matrix)
    # precond_basis_matrix = precond_weights[:,np.newaxis]*basis_matrix
    # precond_train_values = precond_weights[:,np.newaxis]*train_values

    assert np.size(basis_matrix,0) == np.size(basis_matrix,1)
    coef = np.linalg.lstsq(basis_matrix,train_values,rcond=None)[0]
    # coef = np.matmul(train_values, Vinv)
    coef = np.asarray([coef]).T
    coef = np.vstack((coef, ZeroCoeffs))
    # poly.set_coefficients(coef)
    
    # indices = poly.indices
    # recursion_coeffs = np.asarray(poly.recursion_coeffs)
    
    # mesh = UM.generateOrderedGridCenteredAtZero(-0.5,0.5, -0.5,0.5, 0.1, includeOrigin=True)
    # mesh = LP.generateLejaMeshNotCentered(200, min(sigmaX, sigmaY), min(sigmaY,sigmaY), 40, muX, muY)
    # vals1 =(poly.value((train_samples).T))
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(train_samples[:,0], train_samples[:,1], vals1, c='r', marker='.')
    # ax.scatter(train_samples[:,0], train_samples[:,1], train_values, c='k', marker='.')
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

def getNewPDFVal(Px, Py, train_samples, train_values, degree, h):   
    muX = Px
    muY =  Py
    sigmaX = 1 #np.sqrt(h)*g1()
    sigmaY = 1 #np.sqrt(h)*g2()
    
    # train_samples, train_values = getMeshValsThatAreClose(train_samples, train_values, sigmaX, sigmaY, muX, muY)
    mesh1 = LP.mapPointsBack(Px, Py, train_samples, 1/sigmaX, 1/sigmaY)
    lejas1, indices = LP.getLejaSetFromPoints(Px, Py, mesh1, 12, degree)
    lejas = LP.mapPointsBack(Px, Py, lejas1, sigmaX, sigmaY)
    
    pdfNew = []
    Pxs = []
    Pys = []
    for i in range(len(indices)):
        pdfNew.append(train_values[indices[i]])
        Pxs.append(mesh1[indices[i],0])
        Pys.append(mesh1[indices[i],1])
    pdfNew = np.asarray(pdfNew)
    lejas = np.vstack((Pxs, Pys))
    lejas = np.asarray(lejas).T
    
    # pdfNew = UM.generateICPDF(lejas[:,0], lejas[:,1], .1, .1)
    # plt.figure()
    # plt.scatter(train_samples[:,0], train_samples[:,1])
    # plt.plot(Px,Py, '*r')
    # plt.show()
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(lejas[:,0], lejas[:,1], pdfNew, c='r', marker='.')
    integrand = newIntegrand(Px, Py, lejas, h)
    # integrand = np.ones(len(integrand))
    testing = integrand*np.squeeze(pdfNew)

    integral2 = QuadratureByInterpolation(lejas, testing, sigmaX, sigmaY, muX, muY, degree)
    return integral2

# Px = 0
# Py = 0
# h=0.01
# # mesh = UM.generateRandomPoints(-4,4,-4,4,500)  # unordered mesh
# mesh = LP.generateLejaMesh(250, 0.1, 0.1, 50)
# pdf = UM.generateICPDF(mesh[:,0], mesh[:,1], .1, .1)
# # pdf = np.ones(len(pdf))

# value = getNewPDFVal(Px, Py, mesh, pdf, 12, h)




