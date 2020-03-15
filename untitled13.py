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
import Functions as fun
from scipy.interpolate import griddata
import UnorderedMesh as UM
from tqdm import tqdm, trange
import InterpolationPCE as IPCE
from Functions import f1, g1, f2, g2

def getMeshValsThatAreClose(Mesh, pdf, sigmaX, sigmaY, muX, muY):
    MeshToKeep = []
    PdfToKeep = []
    for i in range(len(Mesh)):
        Px = Mesh[i,0]; Py = Mesh[i,1]
        if np.sqrt((Px-muX)**2 + (Py-muY)**2) < 8*max(sigmaX,sigmaY):
            MeshToKeep.append([Px,Py])
            PdfToKeep.append(pdf[i])
    # if len(Mesh)> len(MeshToKeep):        
    #     fig = plt.figure()
    #     ax = Axes3D(fig)
    #     ax.scatter(np.asarray(MeshToKeep)[:,0], np.asarray(MeshToKeep)[:,1], np.asarray(PdfToKeep), c='r', marker='.')
    #     ax.scatter(Mesh[:,0], Mesh[:,1], pdf, c='k', marker='.')
    #     ax.scatter(muX, muY, 0, c='g', marker='*')
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
    coef = np.asarray([coef[0]]).T
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
    if coef[0][0] < 0:
        print("returned 0")
        return [0], poly
    
    return coef[0], poly


def calculateLg(Px,Py,LejaPoints, h):
    scale = (fun.g1(Px, Py)*fun.g2(Px, Py)) * (fun.g1(LejaPoints[:,0], LejaPoints[:,1])*fun.g2(LejaPoints[:,0], LejaPoints[:,1]))
    # print(scale)
    y1 = LejaPoints[:,0]
    y2 = LejaPoints[:,1]
    values = scale*np.exp((1/2)*((Px-y1 +h*fun.f1(y1,y2))**2/(h*fun.g1(Px,Py)**2) + (Py-y2 +h*fun.f2(y1,y2))**2/(h*fun.g2(Px,Py)**2)
            - ((Px-y1 +h*fun.f1(y1, y2))**2/(h*fun.g1(y1,y2)**2) + (Py-y2 +h*fun.f2(y1,y2))**2/(h*fun.g2(y1,y2)**2)))) 
    values2 = scale*np.exp((1/2)*((Px-y1 -h*fun.f1(y1,y2))**2/(h*fun.g1(Px,Py)**2) + (Py-y2 -h*fun.f2(y1,y2))**2/(h*fun.g2(Px,Py)**2)
            - ((Px-y1 -h*fun.f1(y1, y2))**2/(h*fun.g1(y1,y2)**2) + (Py-y2 -h*fun.f2(y1,y2))**2/(h*fun.g2(y1,y2)**2))))
    
    
    return values, values2 

# def calculateLg2(Px,Py,LejaPoints, h):
#     values = np.exp((Px-LejaPoints[:,0] -h*fun.f1(LejaPoints[:,0], LejaPoints[:,1]))**2/(h*fun.g1(Px,Py)**2) + (Py-LejaPoints[:,1] -h*fun.f2(LejaPoints[:,0], LejaPoints[:,1]))**2/(h*fun.g2(Px,Py)**2)
#            - ((Px-LejaPoints[:,0] -h*fun.f1(LejaPoints[:,0], LejaPoints[:,1]))**2/(h*fun.g1(LejaPoints[:,0], LejaPoints[:,1])**2) - (Py-LejaPoints[:,1] +h*fun.f2(LejaPoints[:,0], LejaPoints[:,1]))**2/(h*fun.g2(LejaPoints[:,0], LejaPoints[:,1])**2))) 
#     return values 

def calculateLf(Px,Py,LejaPoints, h):
    y1 = LejaPoints[:,0]
    y2 = LejaPoints[:,1]
    values = np.exp(-(1/2)*((h**2*fun.f1(y1,y2)**2 + 2*h*fun.f1(y1,y2)*(Px-y1))**2/(h*fun.g1(Px,Py)**2) 
                      + (h**2*fun.f2(y1,y2)**2 + 2*h*fun.f2(y1,y2)*(Py-y2))**2/(h*fun.g2(Px,Py)**2)))
    values2 = np.exp(-(1/2)*((h**2*fun.f1(y1,y2)**2 - 2*h*fun.f1(y1,y2)*(Px-y1))**2/(h*fun.g1(Px,Py)**2) 
                        + (h**2*fun.f2(y1,y2)**2 - 2*h*fun.f2(y1,y2)*(Py-y2))**2/(h*fun.g2(Px,Py)**2)))
    return values, values2


def newIntegrand(x1,x2,mesh,h):
    y1 = mesh[:,0]
    y2 = mesh[:,1]
    scale = h*g1(x1,x2)*g2(x1,x2)/(h*g1(y1,y2)*g2(y1,y2))
    val = scale*np.exp(-(h**2*f1(y1,y2)**2+2*h*f1(y1,y2)*(x1-y1))/(2*h*g1(x1,x2)**2) + -(h**2*f2(y1,y2)**2+2*h*f2(y1,y2)*(x2-y2))/(2*h*g2(x1,x2)**2))*np.exp((x1-y1+h*f1(y1,y2))**2/(2*h*g1(x1,x2)**2) - (x1-y1+h *f1(y1,y2))**2/(2*h*g1(y1,y2)**2) + (x2-y2+h*f2(y1,y2))**2/(2*h*g2(x1,x2)**2) - (x2-y2+h* f2(y1,y2))**2/(2*h*g2(y1,y2)**2))
    val = scale*np.exp(-(h**2*f1(y1,y2)**2-2*h*f1(y1,y2)*(x1-y1))/(2*h*g1(x1,x2)**2) + -(h**2*f2(y1,y2)**2-2*h*f2(y1,y2)*(x2-y2))/(2*h*g2(x1,x2)**2))*np.exp((x1-y1-h*f1(y1,y2))**2/(2*h*g1(x1,x2)**2) - (x1-y1-h *f1(y1,y2))**2/(2*h*g1(y1,y2)**2) + (x2-y2-h*f2(y1,y2))**2/(2*h*g2(x1,x2)**2) - (x2-y2-h* f2(y1,y2))**2/(2*h*g2(y1,y2)**2))

    return val

mesh = np.asarray([[1,-1]])
vals = newIntegrand(0,0,mesh,0.01)

# import untitled9 as u9
mesh, weights, poly = IPCE.getLejaQuadratureRule(1, 1, 0, 0)

def getNewPDFVal(Px, Py, train_samples, train_values, degree, h):   
    muX = Px #- h*fun.f1(Px,Py)
    muY =  Py #- h*fun.f2(Px,Py)
    sigmaX = np.sqrt(h)*fun.g1()
    sigmaY = np.sqrt(h)*fun.g2()
    # muX = Px
    # muY = Py
    # sigmaX = 0.1
    # sigmaY = 0.1
    
    mesh1 = LP.mapPointsBack(muX, muY, mesh, sigmaX, sigmaY)
    
    # train_samples, train_values = getMeshValsThatAreClose(train_samples, train_values, sigmaX, sigmaY, muX, muY)
    
    integral = newIntegrand(Px, Py, train_samples, h)
    # testing = integral*np.squeeze(train_values)
    # print(np.max(np.abs(testing-train_values)))

    grid_z2 = griddata(train_samples, np.squeeze(train_values)*integral, mesh1, method='cubic', fill_value=0)
    grid_z1 = griddata(train_samples, np.squeeze(train_values), mesh1, method='cubic', fill_value=0)
    # grid_z0 = griddata(train_samples, Lf, mesh1, method='cubic', fill_value=0)
    
    grid = np.squeeze(grid_z2)

    integral = np.dot(weights,grid)[0]
    
    if integral >10:
        print("Large")
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(train_samples[:,0], train_samples[:,1], np.squeeze(train_values)*integral, c='r', marker='*')
        ax.scatter(mesh1[:,0], mesh1[:,1], grid, c='g', marker='.')
        ax.scatter(train_samples[:,0], train_samples[:,1], np.squeeze(train_values), c='k', marker='*')

    # integral2, poly = QuadratureByInterpolation(train_samples, testing, sigmaX, sigmaY, muX, muY, degree)
    # print(integral-integral2[0])

    # assert np.abs(integral2-integral)<0.1
    return integral



# integral = getNewPDFVal(0, 0, mesh, np.ones((len(mesh),1)), 20, .01)

