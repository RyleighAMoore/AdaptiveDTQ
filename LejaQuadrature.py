
import numpy as np
from scipy.stats import uniform, beta, norm, multivariate_normal
from functools import partial
import numpy as np
import UnorderedMesh as UM
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Functions import f1, f2, g1, g2, GVals
from scipy.interpolate import griddata, interp2d 
from pyopoly1.LejaPoints import getLejaSetFromPoints, mapPointsBack, mapPointsTo, getLejaPoints
from pyopoly1.QuadratureRules import QuadratureByInterpolationND, QuadratureByInterpolation_Simple, QuadratureByInterpolationND_DivideOutGaussian
from pyopoly1.families import HermitePolynomials
from pyopoly1.Scaling import GaussScale
from pyopoly1 import indexing
import math



def newIntegrand(x1,x2,mesh,h):
    '''Calculates the linearization of the Guassian. This newIntegrand times pdf integrated against 
    scaling = GaussScale(2)
    scaling.setMu(np.asarray([[muX,muY]]).T)
    scaling.setSigma(np.asarray([sigmaX,sigmaY])) is used when Quadratic fit fails.'''
    y1 = mesh[:,0]
    y2 = mesh[:,1]
    scale = h*g1(x1,x2)*g2(x1,x2)/(h*g1(y1,y2)*g2(y1,y2))
    val = scale*np.exp(-(h**2*f1(y1,y2)**2+2*h*f1(y1,y2)*(x1-y1))/(2*h*g1(x1,x2)**2) + -(h**2*f2(y1,y2)**2+2*h*f2(y1,y2)*(x2-y2))/(2*h*g2(x1,x2)**2))*np.exp((x1-y1+h*f1(y1,y2))**2/(2*h*g1(x1,x2)**2) - (x1-y1+h *f1(y1,y2))**2/(2*h*g1(y1,y2)**2) + (x2-y2+h*f2(y1,y2))**2/(2*h*g2(x1,x2)**2) - (x2-y2+h* f2(y1,y2))**2/(2*h*g2(y1,y2)**2))
    val = scale*np.exp(-(h**2*f1(y1,y2)**2-2*h*f1(y1,y2)*(x1-y1))/(2*h*g1(x1,x2)**2) + -(h**2*f2(y1,y2)**2-2*h*f2(y1,y2)*(x2-y2))/(2*h*g2(x1,x2)**2))*np.exp((x1-y1-h*f1(y1,y2))**2/(2*h*g1(x1,x2)**2) - (x1-y1-h *f1(y1,y2))**2/(2*h*g1(y1,y2)**2) + (x2-y2-h*f2(y1,y2))**2/(2*h*g2(x1,x2)**2) - (x2-y2-h* f2(y1,y2))**2/(2*h*g2(y1,y2)**2))
    return val


def getMeshValsThatAreClose(Mesh, pdf, sigmaX, sigmaY, muX, muY, numStd = 4):
    MeshToKeep = []
    PdfToKeep = []
    for i in range(len(Mesh)):
        Px = Mesh[i,0]; Py = Mesh[i,1]
        if np.sqrt((Px-muX)**2 + (Py-muY)**2) < numStd*max(sigmaX,sigmaY):
            MeshToKeep.append([Px,Py])
            PdfToKeep.append(pdf[i])
    return np.asarray(MeshToKeep), np.asarray(PdfToKeep)


'''Generate Leja Sample for use in alternative method if needed'''
poly = HermitePolynomials(rho=0)
d=2
k = 40    
ab = poly.recurrence(k+1)
lambdas = indexing.total_degree_indices(d, k)
poly.lambdas = lambdas
lejaPointsFinal, new = getLejaPoints(12, np.asarray([[0,0]]).T, poly, num_candidate_samples=5000, candidateSampleMesh = [], returnIndices = False)

def Test_LejaQuadratureLinearizationOnLejaPoints(mesh, pdf, poly, h, numNodes, step):
    sigmaX=np.sqrt(h)*g1()
    sigmaY=np.sqrt(h)*g2()
    
    newPDF = []
    condNums = []
    countUseMorePoints = 0 # Used to count if we have to revert to alternative procedure
    
    '''Try to Divide out Guassian using quadratic fit'''
    for ii in range(len(mesh)):
        # print('########################',ii/len(mesh)*100, '%')
        muX = mesh[ii,0] 
        muY = mesh[ii,1]
        
        scaling = GaussScale(2)
        scaling.setMu(np.asarray([[muX,muY]]).T)
        scaling.setSigma(np.asarray([sigmaX,sigmaY]))
      
        value, condNum, scaleUsed = QuadratureByInterpolationND_DivideOutGaussian(scaling, h, poly, mesh, np.expand_dims(GVals(muX, muY, mesh, h),1)*pdf)
      
        '''Alternative Method'''
        if math.isnan(condNum) or value <0 or condNum >10: 
            mesh12 = mapPointsBack(muX, muY, lejaPointsFinal, sigmaX, sigmaY)
    
            pdfNew = np.asarray(griddata(mesh, pdf, mesh12, method='cubic', fill_value=0))
            pdfNew[pdfNew < 0] = 10**(-8)
            
            integrand = newIntegrand(muX, muY, mesh12, h)
            testing = np.squeeze(pdfNew)*integrand
            
            value, condNum = QuadratureByInterpolation_Simple(poly, scaling, mesh12, testing)
            value = value*(1/np.sqrt(2))
            countUseMorePoints = countUseMorePoints+1
            
            if value <0:
                value = 10**(-8)

        newPDF.append(value)
        condNums.append(condNum)
        
    print('\n',(countUseMorePoints/len(mesh))*100, "% Used Interpolation*********************************")
    newPDFs = np.asarray(newPDF)
    condNums = np.asarray([condNums]).T
    
    return newPDFs,condNums, mesh


