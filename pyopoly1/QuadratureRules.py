# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 12:44:33 2020

@author: Rylei
"""
import variableTransformations as VT
import numpy as np
import matplotlib.pyplot as plt
import opolynd
from mpl_toolkits.mplot3d import Axes3D
import Scaling
import sys
sys.path.insert(1, r'C:\Users\Rylei\Documents\SimpleDTQ')
from Functions import *
from Scaling import GaussScale
from Plotting import productGaussians2D
import UnorderedMesh as UM
import LejaPoints as LP
from QuadratureUtils import *


def QuadratureByInterpolation1D(poly, scaling, mesh, pdf):
    xCan=VT.map_to_canonical_space(mesh, scaling) 
    V = poly.eval(xCan, range(len(xCan)))
    vinv = np.linalg.inv(V)
    c = np.matmul(vinv, pdf)
    # print(c[0])
    plot=False
    if plot:
        interp = np.matmul(V,c)
        plt.figure()
        plt.plot(mesh, interp,'.')
        plt.plot(mesh, pdf)
    return c[0]
        
def QuadratureByInterpolationND(poly, scaling, mesh, pdf):
    xCan=VT.map_to_canonical_space(mesh, scaling)
    numSamples = len(xCan)                 
    V = opolynd.opolynd_eval(xCan, poly.lambdas[:numSamples,:], poly.ab, poly)
    vinv = np.linalg.inv(V)
    c = np.matmul(vinv, pdf)
    # print(c[0])
    plot = False
    if plot:
        if np.sum(np.abs(vinv[0,:])) > 0:
            interp = np.matmul(V,c)
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(mesh[:,0], mesh[:,1], pdf, c='r', marker='o')
            ax.scatter(mesh[:,0], mesh[:,1], interp, c='k', marker='.')
    return c[0], np.sum(np.abs(vinv[0,:]))


def QuadratureByInterpolationND_FirstStepWithICGaussian(Px,Py, poly, scale0, mesh, h):
    sigmaX = np.sqrt(h)*g1()
    sigmaY = np.sqrt(h)*g2()
    scale = GaussScale(2)
    scale.setMu(np.asarray([[Px,Py]]).T)
    scale.setSigma(np.asarray([sigmaX,sigmaY]))
    
    pdfNew = HVals(Px, Py, mesh, h)

    scaleNew, cfinal = productGaussians2D(scale, scale0)
    
    soln, cond = QuadratureByInterpolationND(poly, scaleNew, mesh, pdfNew)
        
    return soln*cfinal


from QuadratureUtils import GetGaussianPart
from GaussFit import fitGaussian
from QuadraticFit import fitQuad
from scipy.interpolate import griddata
import Functions as fun

# def QuadratureByInterpolationND_DivideOutGaussianExactish(scaling, mesh, pdf, h, poly, fullMesh, fullPDF):
#     # scale1, pdfNew = GetGaussianPart(mesh, pdf, h)
#     # fig = plt.figure()
#     # ax = Axes3D(fig)
#     # ax.scatter(mesh[:,0], mesh[:,1],pdf, c='r', marker='.')

#     # scale1, A, gauss, covPart = fitGaussian(scaling.mu[0][0],scaling.mu[1][0], mesh, pdf)
#     sigmaX = np.sqrt(h)*fun.g1()
#     sigmaY = np.sqrt(h)*fun.g2()
    
#     # constPart = 1/(2*np.pi*sigmaX*sigmaY)*(1/(sigmaX*sigmaY))*(sigmaX*sigmaY)*1/(2*np.pi*sigmaX*sigmaY)
#     # pdfWOConst = pdf/constPart
#     scale1, temp = fitQuad(scaling.mu[0][0],scaling.mu[1][0], mesh, pdf)
   
    
#     mesh2, two = LP.getLejaPoints(12, np.asarray([[0,0]]).T, poly, candidateSampleMesh = [], returnIndices = False)
#     mesh2 = LP.mapPointsBack(scaling.mu[0], scaling.mu[1], mesh2, np.sqrt(scale1.cov[0,0]), np.sqrt(scale1.cov[1,1]))
#     # meshScale1, pdfScale1 = LP.getLejaSetFromPoints(scale1, fullMesh, 6, poly, fullPDF, 0)
    
#     rect = UM.generateOrderedGridCenteredAtZero(-.3, .3, -.3, .3, 0.01, includeOrigin=True)
#     gaussWeight = fun.Gaussian(scale1, rect)
   
   
#     # plt.figure()
#     # plt.scatter(mesh[:,0], mesh[:,1])
#     # plt.scatter(mesh2[:,0], mesh2[:,1], c='red')
#     gauss2 = fun.Gaussian(scale1, mesh2)
#     # pdfOnNewGrid = np.asarray([griddata(fullMesh, fullPDF, mesh2, method='cubic', fill_value=np.min(pdf))])[0]
#     pdfOnNewGrid = GVals(scaling.mu[0][0], scaling.mu[1][0], mesh2, h)*UM.generateICPDF(mesh2[:,0], mesh2[:,1], np.sqrt(h)*fun.g1(),np.sqrt(h)*fun.g2())
    
#     pdf2 = np.expand_dims(pdfOnNewGrid,1) /np.expand_dims(gauss2, axis=1)
    
#     fig = plt.figure()
#     ax = Axes3D(fig)
#     ax.scatter(rect[:,0], rect[:,1],gaussWeight, c='r', marker='.')
#     # ax.scatter(mesh2[:,0], mesh2[:,1],np.ones(len(mesh2)), c='k', marker='o')
#     ax.scatter(mesh2[:,0], mesh2[:,1],pdf2, c='k', marker='o')
#     plt.show()
    
#     # fig = plt.figure()
#     # ax = Axes3D(fig)
#     # ax.scatter(fullMesh[:,0], fullMesh[:,1], fullPDF, c='r', marker='.')
#     # ax.scatter(mesh2[:,0], mesh2[:,1],pdfOnNewGrid, c='k', marker='o')
#     # ax.scatter(mesh[:,0], mesh[:,1], pdf, c='g', marker='.')


    
#     # pdfNew = pdf/np.expand_dims(gauss,1)*np.expand_dims(covPart,1)
#     # print(scaling.mu)
#     # print(np.max(pdfNew), np.min(pdfNew))
#     # fig = plt.figure()
#     # ax = Axes3D(fig)
#     # ax.scatter(mesh[:,0], mesh[:,1], pdfNew, c='r', marker='.')
    
#     # scaleNew, cfinal = productGaussians2D(scale, scaling)
#     value, condNum = QuadratureByInterpolationND(poly, scale1, mesh2, pdf2)
#     print(value, condNum)
#     # v = value*A
#     # if v > 20:
#     #     fig = plt.figure()
#     #     ax = Axes3D(fig)
#     #     ax.scatter(mesh[:,0], mesh[:,1], pdfNew, c='r', marker='.')
    
#     return value[0], condNum


def QuadratureByInterpolationND_DivideOutGaussian(scaling, mesh, pdf, h, poly, fullMesh, fullPDF, ii):
    # scale1, pdfNew = GetGaussianPart(mesh, pdf, h)
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(mesh[:,0], mesh[:,1],pdf, c='r', marker='.')

    # scale1, A, gauss, covPart = fitGaussian(scaling.mu[0][0],scaling.mu[1][0], mesh, pdf)
    sigmaX = np.sqrt(h)*fun.g1()
    sigmaY = np.sqrt(h)*fun.g2()
    
    # constPart = 1/(2*np.pi*sigmaX*sigmaY)*(1/(sigmaX*sigmaY))*(sigmaX*sigmaY)*1/(2*np.pi*sigmaX*sigmaY)
    # pdfWOConst = pdf/constPart
    scale1, temp = fitQuad(scaling.mu[0][0],scaling.mu[1][0], mesh, pdf)
    # scale1 = GaussScale(2)
    # scale1.setMu(scaling.mu)
    # scale1.setSigma(np.asarray([np.sqrt(0.005),np.sqrt(0.005)]))
    
    # print(scale1.mu)
    # print(scale1.cov)
    mesh2, pdfNew = LP.getLejaSetFromPoints(scale1, fullMesh, 12, poly, fullPDF, ii)
    if np.isclose(scale1.cov[0,0],0.005) or np.isclose(scale1.cov[1,1],0.005):
        te=0
    else:
        r=0

    # meshL, two = LP.getLejaPoints(12, np.asarray([[0,0]]).T, poly, candidateSampleMesh = [], returnIndices = False)
    # meshL = LP.mapPointsBack(scaling.mu[0], scaling.mu[1], meshL, np.sqrt(scale1.cov[0,0]), np.sqrt(scale1.cov[1,1]))
    # meshScale1, pdfScale1 = LP.getLejaSetFromPoints(scale1, fullMesh, 6, poly, fullPDF, 0)
    # 
    rect = UM.generateOrderedGridCenteredAtZero(-.3, .3, -.3, .3, 0.01, includeOrigin=True)
    gaussWeight = fun.Gaussian(scale1, rect)
   
   
    # plt.figure()
    # plt.scatter(fullMesh[:,0], fullMesh[:,1])
    # plt.scatter(mesh2[:,0], mesh2[:,1], c='red')
    
    gauss2 = fun.Gaussian(scale1, mesh2)
    pdf2 = pdfNew /np.expand_dims(gauss2, axis=1)
    # print(gauss2)
    if np.max(pdf2) > np.max(fullPDF):
        print("")
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(rect[:,0], rect[:,1],gaussWeight, c='r', marker='.')
    # ax.scatter(mesh2[:,0], mesh2[:,1], pdfNew, c='k', marker='o')
    # plt.show()
    
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(fullMesh[:,0], fullMesh[:,1], fullPDF, c='r', marker='.')
    # ax.scatter(mesh2[:,0], mesh2[:,1],pdfOnNewGrid, c='k', marker='o')
    # ax.scatter(mesh[:,0], mesh[:,1], pdf, c='g', marker='.')


    
    # pdfNew = pdf/np.expand_dims(gauss,1)*np.expand_dims(covPart,1)
    # print(scaling.mu)
    # print(np.max(pdfNew), np.min(pdfNew))
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(mesh[:,0], mesh[:,1], pdfNew, c='r', marker='.')
    
    # scaleNew, cfinal = productGaussians2D(scale, scaling)
    # mesh2 = mesh2[1:,:]
    # pdf2 = pdf2[1:]
    
    
    value, condNum = QuadratureByInterpolationND(poly, scale1, mesh2, pdf2)
    value = pdf2[-1]
    
    # if value > fullPDF[ii]:
    #     print(scale1.cov)
    #     print('\n ', scale1.mu)
        # print(fullPDF[ii], value, condNum)
        # print(' ')
        # print(pdf2)
    # if condNum > 5:
    #     plt.figure()
    #     plt.scatter(fullMesh[:,0], fullMesh[:,1])
    #     plt.scatter(mesh2[:,0], mesh2[:,1], c='red')
    #     plt.scatter(scaling.mu[0], scaling.mu[1], c='g', marker='o')
    #     # plt.scatter(meshL[:,0], meshL[:,1], c='b', marker='*')

    print(value, condNum)
    # v = value*A
    # if v > 20:
    #     fig = plt.figure()
    #     ax = Axes3D(fig)
    #     ax.scatter(mesh[:,0], mesh[:,1], pdfNew, c='r', marker='.')
    assert np.isclose(pdf2[-2], pdf2[-1])
    return value[0], condNum
        

if __name__ == "__main__":
    from families import HermitePolynomials
    import indexing
    import sys
    sys.path.insert(1, r'C:\Users\Rylei\Documents\SimpleDTQ')
    
    # import getPCE as PCE
    # import GenerateLejaPoints as LP
    import UnorderedMesh as UM
    from LejaPoints import getLejaPoints,mapPointsBack


    def fun1D(mesh):
        return 2*mesh**2+4
    
    def fun2D(mesh):
        
        # return np.ones(len(mesh))
        return 2*mesh[:,0]*mesh[:,1]**2 + 4
   
    '''
    1D example of QuadratureByInterpolation
    '''
    H = HermitePolynomials(rho=0)
    mu=0
    sigma=.1
    scaling = np.asarray([[mu, sigma]])
    N=2
    # mesh, w = H.gauss_quadrature(N)
    mesh = np.linspace(-1,1, N)
    pdf = fun1D(mesh)
    QuadratureByInterpolation1D(H, scaling, mesh, pdf)
    ''''''
    
    '''
    2D example of QuadratureByInterpolation
    '''
    # d=2
    # k = 40    
    # ab = H.recurrence(k+1)
    # lambdas = indexing.total_degree_indices(d, k)
    
    # N = 5
    # # x = np.linspace(-5,5, N)
    # mesh, two = getLejaPoints(36, np.asarray([[0,0]]).T, H, lambdas, ab, candidateSampleMesh = [], returnIndices = False)
    # scaling = np.asarray([[0, 1], [0, 1]])
    # pdf = fun2D(mesh)
    
    # QuadratureByInterpolationND(H, scaling, mesh, pdf, lambdas, ab)
    ''''''
    from Functions import g1, g2, f1, f2
    def fun2D(mesh, ic):
        # return np.ones(len(mesh))
        # return np.exp(5*mesh[:,0]**2)
        return UM.generateICPDF(mesh[:,0], mesh[:,1], ic, ic)
    
    def newIntegrand(x1,x2,mesh,h):
        y1 = mesh[:,0]
        y2 = mesh[:,1]
        scale = h*g1(x1,x2)*g2(x1,x2)/(h*g1(y1,y2)*g2(y1,y2))
        val = scale*np.exp(-(h**2*f1(y1,y2)**2+2*h*f1(y1,y2)*(x1-y1))/(2*h*g1(x1,x2)**2) + -(h**2*f2(y1,y2)**2+2*h*f2(y1,y2)*(x2-y2))/(2*h*g2(x1,x2)**2))*np.exp((x1-y1+h*f1(y1,y2))**2/(2*h*g1(x1,x2)**2) - (x1-y1+h *f1(y1,y2))**2/(2*h*g1(y1,y2)**2) + (x2-y2+h*f2(y1,y2))**2/(2*h*g2(x1,x2)**2) - (x2-y2+h* f2(y1,y2))**2/(2*h*g2(y1,y2)**2))
        val = scale*np.exp(-(h**2*f1(y1,y2)**2-2*h*f1(y1,y2)*(x1-y1))/(2*h*g1(x1,x2)**2) + -(h**2*f2(y1,y2)**2-2*h*f2(y1,y2)*(x2-y2))/(2*h*g2(x1,x2)**2))*np.exp((x1-y1-h*f1(y1,y2))**2/(2*h*g1(x1,x2)**2) - (x1-y1-h *f1(y1,y2))**2/(2*h*g1(y1,y2)**2) + (x2-y2-h*f2(y1,y2))**2/(2*h*g2(x1,x2)**2) - (x2-y2-h* f2(y1,y2))**2/(2*h*g2(y1,y2)**2))
        return val
    
    
    import numpy as np
    def productGaussians2D(muX1, muY1, muX2, muY2, sX1, sY1, sX2, sY2):
        cov1 = np.zeros((2,2)); cov1[0,0] = sX1**2; cov1[1,1] = sY1**2
        cov2 = np.zeros((2,2)); cov2[0,0] = sX2**2; cov2[1,1] = sY2**2
        mu1 = np.zeros((2,1)); mu1[0] =muX1; mu1[1] = muY1  
        mu2 = np.zeros((2,1)); mu2[0] =muX2; mu1[1] = muY2 
        
        sigmaNew = np.linalg.inv(np.linalg.inv(cov1)+ np.linalg.inv(cov2))     
        muNew = np.matmul(np.linalg.inv(np.linalg.inv(cov1) + np.linalg.inv(cov2)), np.matmul(np.linalg.inv(cov1),mu1) + np.matmul(np.linalg.inv(cov2),mu2))
        
        c = 1/(np.sqrt(np.linalg.det(2*np.pi*(cov1+cov2))))
        cc = np.matmul(np.matmul(-(1/2)*(mu1-mu2).T, np.linalg.inv(cov1+cov2)),(mu1-mu2))
        cfinal = c*np.exp(cc)
        return muNew, sigmaNew, cfinal[0][0]
    

    '''
    2D example of QuadratureByInterpolation with corrected integrand
    '''
    H = HermitePolynomials(rho=0)
    d=2
    k = 40    
    ab = H.recurrence(k+1)
    lambdas = indexing.total_degree_indices(d, k)
    H.lambdas = lambdas
    
    # x = np.linspace(-5,5, N)
    # mesh2, two = getLejaPoints(230, np.asarray([[0,0]]).T, H, candidateSampleMesh = [], returnIndices = False)
    
    # mesh = mapPointsBack(0, 0, mesh, .05, .05)
    
    # pdf = fun2D(mesh)
    
    # muNew, sigmaNew, cfinal = productGaussians2D(0, 0, 0, 0, 0.1, 0.1, 0.05, 0.05)
    
    # sigmaX = sigmaNew[0,0]
    # sigmaY = sigmaNew[1,1]
    # scaling = np.asarray([[0, sigmaX], [0, sigmaY]])

    # value, condNum = QuadratureByInterpolationND(H, scaling, mesh, pdf)
    # valueFinal = value*cfinal
    # print(valueFinal, condNum)
    ''''''

    
    # def ComputeUpdateIntegralWithCorrection(Px, Py, mesh, correctionVal, H):
    #         mesh = mapPointsBack(Px, Py, mesh, 0.1, 0.1)
           
    #         # plt.figure()
    #         # plt.scatter(mesh[:,0], mesh[:,1])
            
    #         pdf = fun2D(mesh,0.1)*newIntegrand(Px,Py,mesh,0.01) #/UM.generateICPDFShifted(mesh[:,0], mesh[:,1], correctionVal, correctionVal, Px,Py)
            
            
    #         # fig = plt.figure()
    #         # ax = Axes3D(fig)
    #         # ax.scatter(mesh[:,0], mesh[:,1], pdf,  c='r', marker='o')
            
    #         # muNew, sigmaNew, cfinal = productGaussians2D(Px, Py, Px, Py, 0.1, 0.1, correctionVal, correctionVal)
            
    #         # sigmaX = sigmaNew[0,0]
    #         # sigmaY = sigmaNew[1,1]
    #         # muX = muNew[0]
    #         # muY = muNew[1]
    #         scaling = np.asarray([[Px, 0.1], [Py, 0.1]])

    #         value, condNum = QuadratureByInterpolationND(H, scaling, mesh, pdf)
    #         # valueFinal = value*cfinal
    #         # print(valueFinal, condNum)
    #         return value
        
        
    # def ComputeUpdateIntegralWithCorrection(Px, Py, mesh, correctionVal, H):
    #     h=0.01
    #     ic = 0.1
    #     mesh = mapPointsBack(Px, Py, mesh, 0.1, 0.1)
    #     # plt.figure()
    #     # plt.scatter(mesh[:,0], mesh[:,1])
    #     #*newIntegrand(Px,Py,mesh,h)
    #     pdf = fun2D(mesh, ic)*newIntegrand(Px,Py,mesh,h) 
    #     vals = np.cov(mesh.T, aweights = pdf)
    #     # print(vals)
    #     pdf = pdf #/ UM.generateICPDFShifted(mesh[:,0], mesh[:,1], correctionVal, correctionVal,0,0)

    #     # fig = plt.figure()
    #     # ax = Axes3D(fig)
    #     # ax.scatter(mesh[:,0], mesh[:,1], pdf,  c='r', marker='o')
        
    #     muNew, sigmaNew, cfinal = productGaussians2D(0, 0, Px, Py, ic, ic, correctionVal, correctionVal)
    #     # print(muNew)
    #     sigmaX = sigmaNew[0,0]
    #     sigmaY = sigmaNew[1,1]
    #     muX = muNew[0]
    #     muY = muNew[1]
    #     # scaling = np.asarray([[muX, sigmaX], [muY, sigmaY]])
    #     scaling = np.asarray([[Px, 0.1], [Py, 0.1]])

    #     value, condNum = QuadratureByInterpolationND(H, scaling, mesh, pdf)
    #     valueFinal = value*cfinal
    #     # print(valueFinal, condNum)
        
    #     return valueFinal
    
    # mesh2, two = getLejaPoints(60, np.asarray([[0,0]]).T, H, candidateSampleMesh = [], returnIndices = False)
    # mesh = UM.generateOrderedGridCenteredAtZero(-1, 1, -1, 1, 0.05, includeOrigin=True)
    # pdfNew = []
    # for i in range(len(mesh)):
    #     pdfNew.append(np.copy(ComputeUpdateIntegralWithCorrection(mesh[i,0] , mesh[i,1],  mesh2, 0.1, H)))
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(mesh[:,0], mesh[:,1], np.asarray(pdfNew),  c='r', marker='o')
    # ax.scatter(mesh[:,0], mesh[:,1], surfaces[1],  c='k', marker='.')

    # from Functions import *

    # def ComputeUpdateIntegralWithCorrection(Px, Py, mesh, correctionVal, H):
    #     h = 0.01
    #     ic = 0.1
    #     mesh = mapPointsBack(Px, Py, mesh, 0.1, 0.1)
    #     # fig = plt.figure()
    #     # ax = Axes3D(fig)
    #     G = GVals(Px,Py, mesh, h)
    #     pdf = fun2D(mesh, 0.1)*fun2D(mesh, 0.1)*newIntegrand(Px,Py,mesh,h)

    #     # ax.scatter(mesh[:,0], mesh[:,1], G,  c='k', marker='o')
    #     meanX = np.mean(mesh[:,0]*pdf)
    #     meanY = np.mean(mesh[:,1]*pdf)
    #     vals = np.cov(mesh.T, aweights = pdf)
    #     cv1 = np.sqrt(vals[0,0])
    #     cv2 = np.sqrt(vals[1,1])
        
    #     # rv = multivariate_normal([meanX, meanY], [[, 0], [0, .1]])        
    #     # soln_vals = np.asarray([rv.pdf(soln_mesh)]).T
        
    #     pdf = pdf / UM.generateICPDFShifted(mesh[:,0], mesh[:,1], cv1, cv2 ,meanX, meanY)
    #     # print(cv1,cv2)

        
    #     # ax.scatter(mesh[:,0], mesh[:,1], np.log(pdf),  c='r', marker='.')
        
    #     # muNew, sigmaNew, cfinal = productGaussians2D(meanX, meanY, Px, Py, ic, ic, cv1, cv2)
    #     # print(muNew)
    #     # sigmaX = sigmaNew[0,0]
    #     # sigmaY = sigmaNew[1,1]
    #     # muX = muNew[0]
    #     # muY = muNew[1]
    #     scaling =  np.asarray([[meanX, cv1], [meanY, cv2]])
    #     value, condNum = QuadratureByInterpolationND(H, scaling, mesh, pdf)
    #     valueFinal = value
    #     # print(valueFinal, condNum)
        
    #     return valueFinal
    
    # mesh2, two = getLejaPoints(30, np.asarray([[0,0]]).T, H, candidateSampleMesh = [], returnIndices = False)
    # mesh = UM.generateOrderedGridCenteredAtZero(-1, 1, -1, 1, 0.05, includeOrigin=True)
    # pdfNew = []
    
    # for i in range(len(mesh)):
    #     pdfNew.append(np.copy(ComputeUpdateIntegralWithCorrection(mesh[i,0] , mesh[i,1],  mesh2, 0.1, H)))
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(mesh[:,0], mesh[:,1], np.asarray(pdfNew),  c='r', marker='o')
    # # ax.scatter(mesh[:,0], mesh[:,1], surfaces[1],  c='k', marker='.')
    