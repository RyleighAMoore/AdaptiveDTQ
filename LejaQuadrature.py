import GenerateLejaPoints as LP
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

def StepForwardFirstStep_ICofGaussian(mesh, pdf, poly, h, numNodes, icSigma = 0.1):
    sigmaX=np.sqrt(h)*g1()
    sigmaY=np.sqrt(h)*g2()
    sigma = np.sqrt(h)
    
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
        
        scale0 = GaussScale(2)
        scale0.setMu(np.asarray([[0,0]]).T)
        scale0.setSigma(np.asarray([icSigma,icSigma]))
        
        mesh1, indices = getLejaSetFromPoints([muX,muY,sigmaX,sigmaY], mesh, numNodes, poly)
        
        value =  QuadratureByInterpolationND_FirstStepWithICGaussian(muX,muY, poly, scale0, mesh1, h)
        if value <0:
            value = 0
        assert value >=0
        newPDF.append(value)
        
    newPDFs = np.asarray(newPDF)
    condNums = np.asarray([condNums]).T
    return newPDFs


def Test_LejaQuadratureLinearizationOnLejaPoints(mesh, pdf, poly, h, numNodes):
    sigmaX=np.sqrt(h)*g1()
    sigmaY=np.sqrt(h)*g2()
    sigma = np.sqrt(h)
    
    newPDF = []
    condNums = []
    interpErrors = []
    # rv = multivariate_normal([0, 0], [[sigma**2, 0], [0, sigma**2]])
    # pdf = np.asarray([rv.pdf(mesh)]).T
    countUseMorePoints = 0
    for ii in range(len(mesh)):
        # print('########################',ii/len(mesh)*100, '%')
        muX = mesh[ii,0]
        muY = mesh[ii,1]

        # mesh1, pdfNew1 = getMeshValsThatAreClose(mesh, pdf, sigmaX, sigmaY, muX, muY)
        meshTemp = np.delete(mesh, ii, axis=0)
        pdfTemp = np.delete(pdf, ii, axis=0)
        
        mesh1, indices = getLejaSetFromPoints([muX,muY,sigmaX,sigmaY], meshTemp, numNodes, poly)
        # plt.figure()
        # plt.plot(mesh1[:,0], mesh1[:,1], 'or')
        # plt.plot(mesh[:,0], mesh[:,1], '.k')
        # plt.plot(muX, muY, 'ob')
        
        meshTemp = np.vstack(([muX,muY], meshTemp))
        pdfTemp = np.vstack((pdf[ii], pdfTemp))
        pdfNew = []
        Pxs = []
        Pys = []
        # pdfGrid = np.asarray(griddata(mesh, pdf, mesh1, method='cubic', fill_value=0))
        for i in range(len(indices)):
            pdfNew.append(pdfTemp[indices[i]])
            Pxs.append(meshTemp[indices[i],0])
            Pys.append(meshTemp[indices[i],1])
        pdfNew1 = np.asarray(pdfNew)
        mesh1 = np.vstack((Pxs, Pys))
        mesh1 = np.asarray(mesh1).T
        
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # ax.scatter(mesh1[:,0], mesh1[:,1], pdfNew1, c='r', marker='o')
        # ax.scatter(mesh1[:,0], mesh1[:,1], pdfGrid, c='k', marker='.')

        # print(len(mesh1))
        integrand = newIntegrand(muX, muY, mesh1, h)
        testing = np.squeeze(pdfNew1)*integrand
        
        # scaling = np.asarray([[muX, sigmaX], [muY, sigmaY]])
        scaling = GaussScale(2)
        scaling.setMu(np.asarray([[muX,muY]]).T)
        scaling.setSigma(np.asarray([sigmaX,sigmaX]))
      
        pdffull = np.expand_dims(GVals(muX, muY, mesh1, h),1)*pdfNew1
        try:
            value, condNum = QuadratureByInterpolationND_DivideOutGaussian(scaling, mesh1, pdffull, h, poly)
        except:
            print('Using alt')
            # value = float('NaN')
            condNum = 11
            value = -1
        if np.isnan(value) or value <0:
            condNum = 11
        # value, condNum = QuadratureByInterpolationND(poly, scaling, mesh1, testing)
        
        # print(value)
        # print(condNum)
        # condNums.append(condNum)
        # interpErrors.append(maxinterpError)
    
        if condNum >5 or value < 0:
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
            
            # allp, new = getLejaPoints(230, np.asarray([[0,0]]).T, poly, num_candidate_samples=5000, candidateSampleMesh = [], returnIndices = False)
            
            mesh12 = mapPointsBack(muX, muY, allp, sigmaX, sigmaY)
            
            
            # plt.figure()
            # plt.scatter(mesh[:,0], mesh[:,1], c='k', marker='*')
            # plt.scatter(mesh12[:,0], mesh12[:,1], c='r', marker='.')
            # plt.scatter(muX, muY, c='g', marker='.')
    
            pdfNew = np.asarray(griddata(mesh, pdf, mesh12, method='cubic', fill_value=0))
            pdfNew[pdfNew < 0] = 0
            integrand = newIntegrand(muX, muY, mesh12, h)
            testing = np.squeeze(pdfNew)*integrand
            
            value, condNum = value, condNum = QuadratureByInterpolationND(poly, scaling, mesh12, testing)
            # print(value)
            if value<0:
                value= 10**(-10)
                # print('#######################################', muX, muY)
                # plt.figure()
                # plt.scatter(mesh[:,0], mesh[:,1], c='k', marker='*')
                # plt.scatter(mesh12[:,0], mesh12[:,1], c='r', marker='.')
                # plt.scatter(muX, muY, c='g', marker='.')
        # print(condNum)
        # assert value < 20
        newPDF.append(value)
        # interpErrors.append(maxinterpError)
        # condNums.append(condNum)
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
    # ax.scatter(Meshes[-1][:,0], Meshes[-1][:,1], np.log(condnums), c='k', marker='.')
    return newPDFs,condNums, mesh



# def Test_LejaQuadratureLinearizationOnLejaPoint(mesh, pdf, poly, h):
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

#         integrand = newIntegrand(muX, muY, mesh, h)
        
#         testing = np.squeeze(pdf)*integrand
#         # rv = multivariate_normal([0, 0], [[sigma**2, 0], [0, sigma**2]])
#         # pdf = np.asarray([rv.pdf(mesh)]).T
        
#         scaling = np.asarray([[muX, sigmaX], [muY, sigmaY]])
#         value, condNum = QuadratureByInterpolationND(poly, scaling, mesh, testing)
#         # print(value)
#         # print(condNum)
#         condNums.append(condNum)
#         # interpErrors.append(maxinterpError)

#         if condNum > 2 or value < 0:
#             countUseMorePoints = countUseMorePoints+1
#             mesh12, pdfNew1 = getMeshValsThatAreClose(mesh, pdf, sigmaX, sigmaY, muX, muY)

#             mesh12 = mapPointsTo(muX, muY, mesh12, 1/sigmaX, 1/sigmaY)
#             num_leja_samples = 30
#             initial_samples = mesh12
#             numBasis=15
#             allp, new  = getLejaPoints(num_leja_samples, initial_samples.T, poly, num_candidate_samples = 230)
            
#             mesh12 = mapPointsBack(muX,muY, allp, sigmaX, sigmaY)
            
            
#             # plt.figure()
#             # plt.scatter(mesh[:,0], mesh[:,1], c='k', marker='*')
#             # plt.scatter(mesh12[:,0], mesh12[:,1], c='r', marker='.')
#             # plt.scatter(muX, muY, c='g', marker='.')
    
#             pdfNew = np.asarray(griddata(mesh, pdf, mesh12, method='cubic', fill_value=0))
#             pdfNew[pdfNew < 0] = 0
#             integrand = newIntegrand(muX, muY, mesh12, h)
#             testing = np.squeeze(pdfNew)*integrand
            
            
#             value, condNum = value, condNum = QuadratureByInterpolationND(poly, scaling, mesh12, testing)
#             # print(value)
#             if value<0:
#                 value= 10**(-10)
#                 # print('#######################################', muX, muY)
#                 # plt.figure()
#                 # plt.scatter(mesh[:,0], mesh[:,1], c='k', marker='*')
#                 # plt.scatter(mesh12[:,0], mesh12[:,1], c='r', marker='.')
#                 # plt.scatter(muX, muY, c='g', marker='.')
#         # print(condNum)
#         assert value < 20
#         newPDF.append(value)
#         # interpErrors.append(maxinterpError)
#         # condNums.append(condNum)
#     # plt.figure()
#     # plt.scatter(np.reshape(mesh[:,0],-1), np.reshape(mesh[:,1],-1), c=np.reshape(np.log(np.asarray(condNums)),-1), s=300, cmap="seismic", edgecolor="k")
#     # plt.colorbar(label="log(Condition Number)")
#     # plt.show()
    
#     # plt.figure()
#     # plt.loglog((np.asarray(condNums)), np.asarray((interpErrors)), '.')
#     # print(countUseMorePoints/len(mesh))
#     newPDFs = np.asarray(newPDF)
#     condNums = np.asarray([condNums]).T
#     # fig = plt.figure()
#     # ax = Axes3D(fig)
#     # ax.scatter(mesh[:,0], mesh[:,1], newPDF, c='r', marker='.')
    
#     # fig = plt.figure()
#     # ax = Axes3D(fig)
#     # ax.scatter(Meshes[-1][:,0], Meshes[-1][:,1], np.log(condnums), c='k', marker='.')
#     return newPDFs,condNums, mesh
