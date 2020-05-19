import numpy as np
from scipy.stats import uniform, beta, norm
from functools import partial
import numpy as np
from scipy.stats import multivariate_normal
import UnorderedMesh as UM
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Functions import f1, f2, g1, g2, GVals
import sys
sys.path.insert(1, r'C:\Users\Rylei\Documents\SimpleDTQ\pyopoly1')
import QuadratureRules as QR
from scipy.interpolate import griddata, interp2d 
import LejaPoints as LP  
from scipy.interpolate import griddata



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

from LejaPoints import getLejaSetFromPoints, mapPointsBack, mapPointsTo, getLejaPoints
from QuadratureRules import QuadratureByInterpolationND, QuadratureByInterpolationND_FirstStepWithICGaussian, QuadratureByInterpolationND_DivideOutGaussian
from families import HermitePolynomials
from Scaling import GaussScale
import indexing
H = HermitePolynomials(rho=0)
d=2
k = 40    
ab = H.recurrence(k+1)
lambdas = indexing.total_degree_indices(d, k)
H.lambdas = lambdas
allp, new = getLejaPoints(60, np.asarray([[0,0]]).T, H, num_candidate_samples=5000, candidateSampleMesh = [], returnIndices = False)

# def StepForwardFirstStep_ICofGaussian(mesh, pdf, poly, h, numNodes, icSigma = 0.1):
#     sigmaX=np.sqrt(h)*g1()
#     sigmaY=np.sqrt(h)*g2()
#     sigma = np.sqrt(h)
    
#     newPDF = []
#     condNums = []
#     interpErrors = []
#     # rv = multivariate_normal([0, 0], [[sigma**2, 0], [0, sigma**2]])
#     # pdf = np.asarray([rv.pdf(mesh)]).T
#     countUseMorePoints = 0
#     for ii in range(len(mesh)):
#         print('########################',ii/len(mesh)*100, '%')
#         muX = mesh[ii,0]
#         muY = mesh[ii,1]
        
#         scale0 = GaussScale(2)
#         scale0.setMu(np.asarray([[0,0]]).T)
#         scale0.setSigma(np.asarray([icSigma,icSigma]))
        
#         scale = GaussScale(2)
#         scale.setMu(np.asarray([[muX,muY]]).T)
#         scale.setSigma(np.asarray([sigmaX,sigmaY]))
        
#         mesh1, indices = getLejaSetFromPoints(scale, mesh, numNodes, poly, pdf, ii)
        
#         value =  QuadratureByInterpolationND_FirstStepWithICGaussian(muX,muY, poly, scale0, mesh1, h)
#         if value <0:
#             value = 0
#         assert value >=0
#         newPDF.append(value)
        
#     newPDFs = np.asarray(newPDF)
#     condNums = np.asarray([condNums]).T
#     return newPDFs


def Test_LejaQuadratureLinearizationOnLejaPoints(mesh, pdf, poly, h, numNodes):
    sigmaX=np.sqrt(h)*g1()
    sigmaY=np.sqrt(h)*g2()
    sigma = np.sqrt(h)
    
    newPDF = []
    condNums = []
    interpErrors = []
    # plt.figure()
    # rv = multivariate_normal([0, 0], [[sigma**2, 0], [0, sigma**2]])
    # pdf = np.asarray([rv.pdf(mesh)]).T
    countUseMorePoints = 0
    for ii in range(len(mesh)):
        print('########################',ii/len(mesh)*100, '%')
        muX = mesh[ii,0]
        muY = mesh[ii,1]
        
        scale = GaussScale(2)
        scale.setMu(np.asarray([[muX/2,muY/2]]).T)
        scale.setSigma(np.asarray([1,1]))
        
        # mesh12, pdfNew1 = getMeshValsThatAreClose(mesh, pdf, sigmaX, sigmaY, muX, muY)

        mesh1, pdfNew1 = LP.getLejaSetFromPoints(scale, mesh, numNodes, poly, pdf, ii)
        # mesh1 = mesh
        # pdfNew1 = pdf
        
        
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # ax.scatter(mesh1[:,0], mesh1[:,1], pdfNew1, c='r', marker='o')
        # ax.scatter(mesh[:,0], mesh[:,1], pdf, c='k', marker='.')
        # ax.scatter(muX, muY, pdf, c='k', marker='.')

        # integrand = newIntegrand(muX, muY, mesh1, h)
        # testing = np.squeeze(pdfNew1)*integrand
        
        scaling = GaussScale(2)
        scaling.setMu(np.asarray([[muX,muY]]).T)
        scaling.setSigma(np.asarray([sigmaX,sigmaY]))
      
        pdffull = np.expand_dims(GVals(muX, muY, mesh1, h),1)*pdfNew1
        value, condNum = QuadratureByInterpolationND_DivideOutGaussian(scaling, mesh1, pdffull, h, poly, mesh, np.expand_dims(GVals(muX, muY, mesh, h),1)*pdf, ii)

        fullVals = np.expand_dims(GVals(muX, muY, mesh, h),1)*pdf
        # if np.isnan(value) or value <0:
        #     condNum = 11
        # value, condNum = QuadratureByInterpolationND(poly, scaling, mesh1, testing)
        
        # print(value)
        # print(condNum)
        # condNums.append(condNum)
        # interpErrors.append(maxinterpError)
        # plt.scatter(muX, muY, c='k', marker='o')
        # plt.scatter(theScale.mu[0][0], theScale.mu[1][0],c='r', marker='.')
    
        if condNum <-1:
            print(muX,muY)
            countUseMorePoints = countUseMorePoints+1
            mesh12, pdfNew1 = getMeshValsThatAreClose(mesh, pdf, sigmaX, sigmaY, muX, muY)
            # plt.figure()
            # plt.scatter(mesh[:,0], mesh[:,1], c='k', marker='*')
            # plt.scatter(mesh12[:,0], mesh12[:,1], c='r', marker='.')
            # plt.scatter(muX, muY, c='g', marker='.')
            
            mesh12 = mapPointsTo(muX, muY, mesh12, 1/sigmaX, 1/sigmaY)
            num_leja_samples = numNodes
            initial_samples = mesh12
            numBasis=numNodes
            initial_samples = np.asarray([[0,0]])
            
            # allp, new = getLejaPoints(12, np.asarray([[0,0]]).T, poly, num_candidate_samples=5000, candidateSampleMesh = [], returnIndices = False)
            
            mesh12 = mapPointsBack(muX, muY, allp, sigmaX/2, sigmaY/2)
            
            
            # plt.figure()
            # plt.scatter(mesh[:,0], mesh[:,1], c='k', marker='*')
            # plt.scatter(mesh12[:,0], mesh12[:,1], c='r', marker='.')
            # plt.scatter(muX, muY, c='g', marker='.')
    
            pdfNew = np.asarray(griddata(mesh, pdf, mesh12, method='cubic', fill_value=0))
            pdfNew[pdfNew < 0] = 0
            integrand = newIntegrand(muX, muY, mesh12, h)
            testing = np.squeeze(pdfNew)*integrand
            
            value, condNum = QuadratureByInterpolationND(poly, scaling, mesh12, testing)
            value = value[0]
           
        # print(condNum)
        # assert value < 20
        newPDF.append(value)
        # interpErrors.append(maxinterpError)
        condNums.append(condNum)
    # plt.figure()
    # plt.scatter(np.reshape(mesh[:,0],-1), np.reshape(mesh[:,1],-1), c=np.reshape(np.log(np.asarray(condNums)),-1), s=300, cmap="seismic", edgecolor="k")
    # plt.colorbar(label="log(Condition Number)")
    # plt.show()
    
    # plt.figure()
    # plt.loglog((np.asarray(condNums)), np.asarray((interpErrors)), '.')
    print(countUseMorePoints/len(mesh), "*********************************")
    newPDFs = np.asarray(newPDF)
    condNums = np.asarray([condNums]).T
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(mesh[:,0], mesh[:,1], newPDF, c='r', marker='.')
    
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(mesh[:,0], mesh[:,1], np.log(condNums), c='k', marker='.')
    # plt.figure()
    # plt.scatter(mesh[:,0], mesh[:,1],c= condNums)
    # plt.colorbar()
    # plt.show()
    return newPDFs,condNums, mesh



