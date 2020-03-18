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
import LejaQuadrature as LQ
from Functions import f1, g1, f2, g2

def getMeshValsThatAreClose(Mesh, pdf, sigmaX, sigmaY, muX, muY):
    MeshToKeep = []
    PdfToKeep = []
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

    # train_values = np.log(train_values)
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
    coef = np.linalg.lstsq(basis_matrix,train_values,rcond=None)[0]
    # ceof = np.exp(coef)
    coef = np.vstack((coef, ZeroCoeffs))
    poly.set_coefficients(coef)
    
    indices = poly.indices
    recursion_coeffs = np.asarray(poly.recursion_coeffs)
    
    # mesh = UM.generateOrderedGridCenteredAtZero(-0.5,0.5, -0.5,0.5, 0.1, includeOrigin=True)
    # mesh = LP.generateLejaMeshNotCentered(200, min(sigmaX, sigmaY), min(sigmaY,sigmaY), 40, muX, muY)
    vals1 =(poly.value((train_samples).T))
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
        
    
    return coef[0], poly


def newIntegrand(x1,x2,mesh,h):
    y1 = mesh[:,0]
    y2 = mesh[:,1]
    scale = h*g1(x1,x2)*g2(x1,x2)/(h*g1(y1,y2)*g2(y1,y2))
    val = scale*np.exp(-(h**2*f1(y1,y2)**2+2*h*f1(y1,y2)*(x1-y1))/(2*h*g1(x1,x2)**2) + -(h**2*f2(y1,y2)**2+2*h*f2(y1,y2)*(x2-y2))/(2*h*g2(x1,x2)**2))*np.exp((x1-y1+h*f1(y1,y2))**2/(2*h*g1(x1,x2)**2) - (x1-y1+h *f1(y1,y2))**2/(2*h*g1(y1,y2)**2) + (x2-y2+h*f2(y1,y2))**2/(2*h*g2(x1,x2)**2) - (x2-y2+h* f2(y1,y2))**2/(2*h*g2(y1,y2)**2))
    val = scale*np.exp(-(h**2*f1(y1,y2)**2-2*h*f1(y1,y2)*(x1-y1))/(2*h*g1(x1,x2)**2) + -(h**2*f2(y1,y2)**2-2*h*f2(y1,y2)*(x2-y2))/(2*h*g2(x1,x2)**2))*np.exp((x1-y1-h*f1(y1,y2))**2/(2*h*g1(x1,x2)**2) - (x1-y1-h *f1(y1,y2))**2/(2*h*g1(y1,y2)**2) + (x2-y2-h*f2(y1,y2))**2/(2*h*g2(x1,x2)**2) - (x2-y2-h* f2(y1,y2))**2/(2*h*g2(y1,y2)**2))

    return val
    
    

# mesh = np.asarray([[1,-1]])
# vals = newIntegrand(0,0,mesh,0.01)

# import untitled9 as u9
import LejaQuadrature as LQ 
def getNewPDFVal(Px, Py, train_samples, train_values, degree, h):   
    muX = Px #- h*fun.f1(Px,Py)
    muY =  Py #- h*fun.f2(Px,Py)
    sigmaX = np.sqrt(h)*fun.g1()
    sigmaY = np.sqrt(h)*fun.g2()
    
    lejaPointsShifted = LP.mapPointsBack(muX, muY, lejaPoints, sigmaX, sigmaY)
    train_samplesO = np.copy(train_samples)
    train_valuesO = np.copy(train_values)
    
    train_samples, train_values = getMeshValsThatAreClose(train_samples, train_values, sigmaX, sigmaY, muX, muY)
    
    integrandVals = newIntegrand(Px, Py, train_samples, h)

    fullIntegrand = np.asarray(np.squeeze(train_values)*integrandVals)
    fullIntegrand = fullIntegrand.reshape((len(fullIntegrand),1))
    # print(np.max(np.abs(testing-train_values)))

    # grid_z2 = griddata(train_samples, np.squeeze(train_values), lejaPointsShifted, method='cubic', fill_value=0)
    grid_z2 = griddata(train_samples, fullIntegrand, lejaPointsShifted, method='cubic', fill_value=10**(-20))
    if min(grid_z2) <=0:
        grid_z2[grid_z2<0] = 10**(-20)

    # grid_z2 = np.exp(grid_z2)

    integralSoln = np.dot(weights,grid_z2)[0]
    
    # integralSoln2, poly = QuadratureByInterpolation(train_samples, fullIntegrand, sigmaX, sigmaY, muX, muY, degree)
    integralSoln2=0
    
    # valExtrap = checkIfExtrapolation(train_samples, train_values, lejaPointsShifted)
    
    # if abs(integralSoln - integralSoln2) > 0.3:
    #      plt.figure()
    #      plt.scatter(train_samplesO[:,0], train_samplesO[:,1], label ='All Known samples')
    #      plt.scatter(train_samples[:,0], train_samples[:,1], label='Inperpolation Known samples')
    #      plt.scatter(lejaPointsShifted[:,0], lejaPointsShifted[:,1], label='Leja point locations')
    #      plt.scatter(Px, Py, label ='Current point')

    #      plt.legend()

    # if integralSoln2 > 10:
    #     print("Large")
    #     fig = plt.figure()
    #     ax = Axes3D(fig)
    #     ax.scatter(train_samples[:,0], train_samples[:,1], np.squeeze(train_values)*integral, c='r', marker='*')
    #     # ax.scatter(mesh1[:,0], mesh1[:,1], grid, c='g', marker='.')
    #     ax.scatter(train_samples[:,0], train_samples[:,1], np.squeeze(train_values), c='k', marker='*')
    if integralSoln < 0:
        integralSoln = max(np.asarray([0]), integralSoln)
        # integralSoln2 = max(np.asarray([0]), integralSoln2)
    # print(integralSoln)

    # print(integralSoln2)

    # assert np.abs(integral2-integral)<0.1
    return integralSoln, integralSoln2


# import pickle
# pickle_in = open("C:/Users/Rylei/Documents/SimpleDTQ-LejaMesh.p","rb")
# mesh = pickle.load(pickle_in)

# Mesh = LP.generateLejaMesh(250, .1, .1, 50)
# mesh = Mesh[:250, :250]

# pdf = UM.generateICPDF(mesh[:,0], mesh[:,1], .1, .1)
# import MeshUpdates2D as MU
# from scipy.spatial import Delaunay
# tri = Delaunay(mesh, incremental=True)

# # Mesh, Pdf, triangulation, changedBool =  MU.addPointsToBoundary(mesh, pdf, tri, 0.01, 0.01)

# plt.figure()
# plt.scatter(Mesh[:,0], Mesh[:,1],  label = 'New')
# plt.scatter(mesh[:,0], mesh[:,1], c='r', label = 'Old')
# plt.legend()


lejaPoints, weights = LQ.getLejaQuadratureRule(1, 1, 0, 0)

# points = []
# diffs = []
# extraps = []
# integral2s = []
# for i in range(len(mesh)):
#     # integral, integral2, extrap = getNewPDFVal(mesh[i,0], mesh[i,1] , Mesh, np.ones((len(Mesh),1)), 55, .01)
#     integral, integral2 = getNewPDFVal(mesh[i,0], mesh[i,1] , mesh, pdf, 55, .01)

#     integral2s.append(integral)
#     # print(abs(integral2-integral))
#     diffs.append(integral-1)
#     if np.abs(integral -1) >0.02:
#         points.append([mesh[i,0], mesh[i,1]])
        
        
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(mesh[:,0], mesh[:,1], np.asarray(diffs), c='r', marker='*')
 

# points = np.asarray(points)
# plt.figure()
# plt.scatter(mesh[:,0], mesh[:,1], c='r', label = 'Accurate')
# plt.scatter(points[:,0], points[:,1],  label = 'Inaccurate')
# plt.legend()

# # extraps = np.asarray(extraps)

# # plt.figure()
# # plt.scatter(mesh[:,0], mesh[:,1], c='r', label = 'Accurate')
# # plt.scatter(extraps[:,0], extraps[:,1],  label = 'Inaccurate')
# # plt.legend()

# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(mesh[:,0], mesh[:,1], np.asarray(integral2s), c='r', marker='*')
# ax.scatter(Mesh[:,0], Mesh[:,1], pdf, c='k', marker='*')


