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

   
def getLejaQuadratureRule(sigmaX, sigmaY, muX, muY):
    degree = 20
    poly = PCE.generatePCE(degree+10, muX=0, muY=0, sigmaX=1, sigmaY=1)
    
    num_leja_samples = 230
    initial_samples = np.asarray([[muX],[muY]])
    # train_samples = LP.generateLejaMeshNotCentered(num_leja_samples, sigmaX, sigmaY, degree,muX,muY)
    train_samples, mesh2 = LP.getLejaPointsWithStartingPoints([muX,muY,sigmaX,sigmaY], num_leja_samples, 5000, poly)

    poly = PCE.generatePCE(degree, muX=muX, muY=muY, sigmaX=sigmaX, sigmaY=sigmaY)
    
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

# train_samples1, weights1 = getLejaQuadratureRule(0.1, 0.1 ,1,1)
# print(np.sum(weights1),1)
# train_samples, weights = getLejaQuadratureRule(0.1, 0.1,0,0)
# print(np.sum(weights),1)

    
# Aa1 = np.matmul(weights1, np.ones(len(train_samples1))) # should be 1
# Aa = np.matmul(weights, np.ones(len(train_samples))) 



def newIntegrand(x1,x2,mesh,h):
    y1 = mesh[:,0]
    y2 = mesh[:,1]
    scale = h*g1(x1,x2)*g2(x1,x2)/(h*g1(y1,y2)*g2(y1,y2))
    val = scale*np.exp(-(h**2*f1(y1,y2)**2+2*h*f1(y1,y2)*(x1-y1))/(2*h*g1(x1,x2)**2) + -(h**2*f2(y1,y2)**2+2*h*f2(y1,y2)*(x2-y2))/(2*h*g2(x1,x2)**2))*np.exp((x1-y1+h*f1(y1,y2))**2/(2*h*g1(x1,x2)**2) - (x1-y1+h *f1(y1,y2))**2/(2*h*g1(y1,y2)**2) + (x2-y2+h*f2(y1,y2))**2/(2*h*g2(x1,x2)**2) - (x2-y2+h* f2(y1,y2))**2/(2*h*g2(y1,y2)**2))
    val = scale*np.exp(-(h**2*f1(y1,y2)**2-2*h*f1(y1,y2)*(x1-y1))/(2*h*g1(x1,x2)**2) + -(h**2*f2(y1,y2)**2-2*h*f2(y1,y2)*(x2-y2))/(2*h*g2(x1,x2)**2))*np.exp((x1-y1-h*f1(y1,y2))**2/(2*h*g1(x1,x2)**2) - (x1-y1-h *f1(y1,y2))**2/(2*h*g1(y1,y2)**2) + (x2-y2-h*f2(y1,y2))**2/(2*h*g2(x1,x2)**2) - (x2-y2-h* f2(y1,y2))**2/(2*h*g2(y1,y2)**2))
    return val


def getMeshValsThatAreClose(Mesh, pdf, sigmaX, sigmaY, muX, muY, numStd = 4):
    MeshToKeep = []
    PdfToKeep = []
    distances = np.sqrt((Mesh[:,0]-muX)**2 + (Mesh[:,1]-muY)**2)
    
    for i in range(len(Mesh)):
        Px = Mesh[i,0]; Py = Mesh[i,1]
        if np.sqrt((Px-muX)**2 + (Py-muY)**2) < numStd*max(sigmaX,sigmaY):
            MeshToKeep.append([Px,Py])
            PdfToKeep.append(pdf[i])
    # if len(Mesh)> len(MeshToKeep):        
    #     fig = plt.figure()
    #     ax = Axes3D(fig)
    #     ax.scatter(np.asarray(MeshToKeep)[:,0], np.asarray(MeshToKeep)[:,1], np.asarray(PdfToKeep), c='r', marker='.')
    #     # ax.scatter(Mesh[:,0], Mesh[:,1], pdf, c='k', marker='.')
    #     ax.scatter(muX, muY, 0, c='g', marker='*')
    return np.asarray(MeshToKeep), np.asarray(PdfToKeep)

indices = compute_hyperbolic_indices(2,20,1.0)
import chaospy
def QuadratureByInterpolation(train_samples, train_values, sigmaX, sigmaY, muX, muY, degree):
    poly = PCE.generatePCE(degree, muX=muX, muY=muY, sigmaX=sigmaX, sigmaY=sigmaY)
    distribution = chaospy.J(chaospy.Normal(muX, sigmaX), chaospy.Normal(muY, sigmaY))
    poly = chaospy.orth_ttr(20, distribution)
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
    # basis_matrix = poly(train_samples.T)
    basis_matrix = poly(train_samples[:,0], train_samples[:,1]).T

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
from LejaQuadrature import getLejaQuadratureRule, getMeshValsThatAreClose, newIntegrand, getNewPDFVal, QuadratureByInterpolation
import numpy as np
from scipy.stats import multivariate_normal
from math import isclose
from scipy.stats import norm
from pyapprox.variables import IndependentMultivariateRandomVariable
from pyapprox.variable_transformations import AffineRandomVariableTransformation
from pyapprox.multivariate_polynomials import PolynomialChaosExpansion, define_poly_options_from_variable_transformation
from pyapprox.indexing import compute_hyperbolic_indices, tensor_product_indices,compute_tensor_product_level_indices
import GenerateLejaPoints as LP
from GenerateLejaPoints import getLejaSetFromPoints, getLejaPoints, mapPointsBack, mapPointsTo
import UnorderedMesh as UM
import numpy as np
import matplotlib.pyplot as plt
from Functions import g1, g2
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy.spatial import Delaunay
def Test_LejaQuadratureLinearizationOnLejaPoints_Slow(mesh, pdf):
    h = 0.01
    sigmaX=np.sqrt(h)*g1()
    sigmaY=np.sqrt(h)*g2()
    sigma = np.sqrt(h)

    # mesh = LP.generateLejaMesh(230, sigmaX, sigmaY, 20)
    
    newPDF = []
    condNums = []
    # rv = multivariate_normal([0, 0], [[sigma**2, 0], [0, sigma**2]])
    # pdf = np.asarray([rv.pdf(mesh)]).T
    countUseMorePoints = 0
    for ii in range(len(mesh)):
        print('########################',ii/len(mesh)*100, '%')
        muX = mesh[ii,0]
        muY = mesh[ii,1]

        # mesh1, pdfNew1 = getMeshValsThatAreClose(mesh, pdf, sigmaX, sigmaY, muX, muY)
        meshTemp = np.delete(mesh, ii, axis=0)
        pdfTemp = np.delete(pdf, ii, axis=0)
        
        mesh1, indices = getLejaSetFromPoints(muX, muY, meshTemp, 130, 20, sigmaX, sigmaY)
        meshTemp = np.vstack(([muX,muY], meshTemp))
        pdfTemp = np.vstack((pdf[ii], pdfTemp))
        pdfNew = []
        Pxs = []
        Pys = []
        for i in range(len(indices)):
            pdfNew.append(pdfTemp[indices[i]])
            Pxs.append(meshTemp[indices[i],0])
            Pys.append(meshTemp[indices[i],1])
        pdfNew1 = np.asarray(pdfNew)
        mesh1 = np.vstack((Pxs, Pys))
        mesh1 = np.asarray(mesh1).T
        
        
        # print(len(mesh1))
        integrand = newIntegrand(muX, muY, mesh1, h)
        testing = np.squeeze(pdfNew1)*integrand
        
        value, condNum= QuadratureByInterpolation(mesh1, testing, sigmaX, sigmaY, muX, muY, 20)
        # print(value)
        if condNum > 2 or value < 0:
            countUseMorePoints = countUseMorePoints+1
            mesh12, pdfNew1 = getMeshValsThatAreClose(mesh, pdf, sigmaX, sigmaY, muX, muY)
            
            needPoints = 130
            if len(mesh12) < needPoints:
                mesh12 = mapPointsTo(muX, muY, mesh12, 1/sigmaX, 1/sigmaY)
                num_leja_samples = 130
                initial_samples = mesh12
                numBasis=20
                allp, new = LP.getLejaPoints(num_leja_samples, initial_samples.T,numBasis, num_candidate_samples = 230, dimensions=2, defCandidateSamples=False, candidateSampleMesh = [], returnIndices = False)
                mesh12 = mapPointsBack(muX,muY, allp, sigmaX, sigmaY)
            
            
            # plt.figure()
            # plt.scatter(mesh[:,0], mesh[:,1], c='k', marker='*')
            # plt.scatter(mesh12[:,0], mesh12[:,1], c='r', marker='.')
            # plt.scatter(muX, muY, c='g', marker='.')
    
            pdfNew = np.asarray(griddata(mesh, pdf, mesh12, method='cubic', fill_value=0))
            pdfNew[pdfNew < 0] = 0
            integrand = newIntegrand(muX, muY, mesh12, h)
            testing = np.squeeze(pdfNew)*integrand
            
            
            value, condNum= QuadratureByInterpolation(mesh12, testing, sigmaX, sigmaY, muX, muY, 20)
            # print(value)
            if value<0:
                value= [10**(-10)]
                # print('#######################################', muX, muY)
                # plt.figure()
                # plt.scatter(mesh[:,0], mesh[:,1], c='k', marker='*')
                # plt.scatter(mesh12[:,0], mesh12[:,1], c='r', marker='.')
                # plt.scatter(muX, muY, c='g', marker='.')
        
        newPDF.append(value)
        condNums.append(condNum)
    # print(countUseMorePoints)
    newPDF = np.asarray(newPDF)
    condNums = np.asarray([condNums]).T
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(mesh[:,0], mesh[:,1], newPDF, c='r', marker='.')
    
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(mesh1[:,0], mesh1[:,1], testing, c='k', marker='.')
    return newPDF,condNums, mesh

from numba import jit
# @jit(parallel=True)
def Test_LejaQuadratureLinearizationOnLejaPoints(mesh, pdf, poly):
    h = 0.01
    sigmaX=np.sqrt(h)*g1()
    sigmaY=np.sqrt(h)*g2()
    sigma = np.sqrt(h)

    # mesh = LP.generateLejaMesh(230, sigmaX, sigmaY, 20)
    
    newPDF = []
    condNums = []
    interpErrors = []
    # rv = multivariate_normal([0, 0], [[sigma**2, 0], [0, sigma**2]])
    # pdf = np.asarray([rv.pdf(mesh)]).T
    countUseMorePoints = 0
    for ii in range(len(mesh)):
        print('########################',ii/len(mesh)*100, '%')
        muX = mesh[ii,0]
        muY = mesh[ii,1]

        # mesh1, pdfNew1 = getMeshValsThatAreClose(mesh, pdf, sigmaX, sigmaY, muX, muY)
        meshTemp = np.delete(mesh, ii, axis=0)
        pdfTemp = np.delete(pdf, ii, axis=0)
        
        mesh1, indices = getLejaSetFromPoints([0,0,sigmaX,sigmaY], meshTemp, 130, poly)
        meshTemp = np.vstack(([muX,muY], meshTemp))
        pdfTemp = np.vstack((pdf[ii], pdfTemp))
        pdfNew = []
        Pxs = []
        Pys = []
        for i in range(len(indices)):
            pdfNew.append(pdfTemp[indices[i]])
            Pxs.append(meshTemp[indices[i],0])
            Pys.append(meshTemp[indices[i],1])
        pdfNew1 = np.asarray(pdfNew)
        mesh1 = np.vstack((Pxs, Pys))
        mesh1 = np.asarray(mesh1).T
        
        
        # print(len(mesh1))
        integrand = newIntegrand(muX, muY, mesh1, h)
        testing = np.squeeze(pdfNew1)*integrand
        
        value, condNum = QuadratureByInterpolation(mesh1, testing, sigmaX, sigmaY, muX, muY, 20)
        # print(value)
        print(condNum)
        condNums.append(condNum)
        # interpErrors.append(maxinterpError)

        if condNum > 2 or value < 0:
            countUseMorePoints = countUseMorePoints+1
            mesh12, pdfNew1 = getMeshValsThatAreClose(mesh, pdf, sigmaX, sigmaY, muX, muY)

            mesh12 = mapPointsTo(muX, muY, mesh12, 1/sigmaX, 1/sigmaY)
            num_leja_samples = 130
            initial_samples = mesh12
            numBasis=15
            allp, new  = LP.getLejaPoints(num_leja_samples, initial_samples.T, poly, num_candidate_samples = 230)
            
            mesh12 = mapPointsBack(muX,muY, allp, sigmaX, sigmaY)
            
            
            # plt.figure()
            # plt.scatter(mesh[:,0], mesh[:,1], c='k', marker='*')
            # plt.scatter(mesh12[:,0], mesh12[:,1], c='r', marker='.')
            # plt.scatter(muX, muY, c='g', marker='.')
    
            pdfNew = np.asarray(griddata(mesh, pdf, mesh12, method='cubic', fill_value=0))
            pdfNew[pdfNew < 0] = 0
            integrand = newIntegrand(muX, muY, mesh12, h)
            testing = np.squeeze(pdfNew)*integrand
            
            
            value, condNum = QuadratureByInterpolation(mesh12, testing, sigmaX, sigmaY, muX, muY, 20)
            # print(value)
            if value<0:
                value= [10**(-10)]
                # print('#######################################', muX, muY)
                # plt.figure()
                # plt.scatter(mesh[:,0], mesh[:,1], c='k', marker='*')
                # plt.scatter(mesh12[:,0], mesh12[:,1], c='r', marker='.')
                # plt.scatter(muX, muY, c='g', marker='.')
        print(condNum)
        assert value[0] < 20
        newPDF.append(value)
        # interpErrors.append(maxinterpError)
        # condNums.append(condNum)
    # plt.figure()
    # plt.scatter(np.reshape(mesh[:,0],-1), np.reshape(mesh[:,1],-1), c=np.reshape(np.log(np.asarray(condNums)),-1), s=300, cmap="seismic", edgecolor="k")
    # plt.colorbar(label="log(Condition Number)")
    # plt.show()
    
    # plt.figure()
    # plt.loglog((np.asarray(condNums)), np.asarray((interpErrors)), '.')
    # print(countUseMorePoints/len(mesh))
    newPDF = np.asarray(newPDF)
    condNums = np.asarray([condNums]).T
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(mesh[:,0], mesh[:,1], newPDF, c='r', marker='.')
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(Meshes[-1][:,0], Meshes[-1][:,1], np.log(condnums), c='k', marker='.')
    return newPDF,condNums, mesh
