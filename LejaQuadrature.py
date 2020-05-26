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
from QuadratureRules import QuadratureByInterpolationND, QuadratureByInterpolationND_DivideOutGaussian
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
import math

def Test_LejaQuadratureLinearizationOnLejaPoints(mesh, pdf, poly, h, numNodes, step):
    sigmaX=np.sqrt(h)*g1()
    sigmaY=np.sqrt(h)*g2()
    sigma = np.sqrt(h)
    
    newPDF = []
    condNums = []
    interpErrors = []
    # plt.figure()
    Lvals00 = []
    Lvals11 = []
    Lvals01 = []
    Lvals10 = []
    muValsX = []
    muValsY = []
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

        # mesh1, pdfNew1 = LP.getLejaSetFromPoints(scale, mesh, numNodes, poly, pdf, ii)
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
      
        # pdffull = np.expand_dims(GVals(muX, muY, mesh1, h),1)*pdfNew1
        value, condNum, scaleUsed = QuadratureByInterpolationND_DivideOutGaussian(scaling, mesh, pdf, h, poly, mesh, np.expand_dims(GVals(muX, muY, mesh, h),1)*pdf, ii,step)
        # Lvals00.append(scaleUsed.cov[0,0])
        # Lvals11.append(scaleUsed.cov[1,1])
        # Lvals10.append(scaleUsed.cov[1,0])
        # Lvals01.append(scaleUsed.cov[0,1])
        # muValsX.append(scaleUsed.mu[0][0])
        # muValsY.append(scaleUsed.mu[1][0])
        
        
        if math.isnan(condNum) or value <0 or condNum >10:
            print(value,condNum)
            
            # mesh12, pdfNew1 = getMeshValsThatAreClose(mesh, pdf, sigmaX, sigmaY, muX, muY)
            # plt.figure()
            # plt.scatter(mesh[:,0], mesh[:,1], c='k', marker='*')
            # plt.scatter(mesh12[:,0], mesh12[:,1], c='r', marker='.')
            # plt.scatter(muX, muY, c='g', marker='.')
            
            # mesh12 = mapPointsTo(muX, muY, mesh, 1/sigmaX, 1/sigmaY)
            # num_leja_samples = numNodes
            # initial_samples = mesh12
            # numBasis=numNodes
            # initial_samples = np.asarray([[0,0]])
            
            lejaPointsFinal, new = LP.getLejaPoints(12, np.asarray([[0,0]]).T, poly, num_candidate_samples=5000, candidateSampleMesh = [], returnIndices = False)
            # mesh12, newLeja = LP.getLejaPointsWithStartingPoints(scaling, 130, 1000, poly, neighbors=[0,[]])
            mesh12 = mapPointsBack(muX, muY, lejaPointsFinal, sigmaX, sigmaY)

            
            # plt.figure()
            # plt.scatter(mesh[:,0], mesh[:,1], c='k', marker='*')
            # plt.scatter(mesh12[:,0], mesh12[:,1], c='r', marker='.')
            # plt.scatter(muX, muY, c='g', marker='.')
    
            pdfNew = np.asarray(griddata(mesh, pdf, mesh12, method='cubic', fill_value=0))
            pdfNew[pdfNew < 0] = 10**(-8)
            
            integrand = newIntegrand(muX, muY, mesh12, h)
            testing = np.squeeze(pdfNew)*integrand
            
            value, condNum = QR.QuadratureByInterpolation_Simple(poly, scaling, mesh12, testing)
            value = value*(1/np.sqrt(2))
            countUseMorePoints = countUseMorePoints+1
            print(value,condNum, "++++++++++++++++++++++++++++++++++")
            if value <0:
                value = 10**(-8)

        
        print(value,condNum)
        # assert value < 20
        newPDF.append(value)
        # interpErrors.append(maxinterpError)
        condNums.append(condNum)
        
    # plt.figure()
    # plt.scatter(muValsX, muValsY, c=Lvals00, s=200, cmap="seismic", edgecolor="k")
    # plt.colorbar()
    # plt.title('L[0,0]')
    # plt.show()
    
    # plt.figure()
    # plt.scatter(muValsX, muValsY, c=Lvals11, s=200, cmap="seismic", edgecolor="k")
    # plt.colorbar()
    # plt.title('L[1,1]')
    # plt.show()
    
    # plt.figure()
    # plt.scatter(muValsX,muValsY, c=Lvals10, s=200, cmap="seismic", edgecolor="k")
    # plt.colorbar()
    # plt.title('L[1,0]')
    # plt.show()
    
    # plt.figure()
    # plt.scatter(muValsX,muValsY, c=Lvals01, s=200, cmap="seismic", edgecolor="k")
    # plt.colorbar()
    # plt.title('L[0,1]')
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



# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(Meshes[-1][:,0], Meshes[-1][:,1], np.expand_dims(pdffull, axis=1), c='r', marker='o')
