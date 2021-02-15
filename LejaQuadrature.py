
import numpy as np
from scipy.stats import uniform, beta, norm, multivariate_normal
from functools import partial
import numpy as np
import UnorderedMesh as UM
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Functions import drift, diff, GVals, Gaussian, G
from scipy.interpolate import griddata, interp2d 
from pyopoly1.LejaPoints import getLejaSetFromPoints, getLejaPoints
from pyopoly1.QuadratureRules import QuadratureByInterpolationND, QuadratureByInterpolation_Simple, QuadratureByInterpolationND_DivideOutGaussian
from pyopoly1.families import HermitePolynomials
from pyopoly1.Scaling import GaussScale
from pyopoly1 import indexing
import math
from pyopoly1 import variableTransformations as VT




def newIntegrand(x1,x2,mesh,h):
    '''Calculates the linearization of the Guassian. This newIntegrand times pdf integrated against 
    scaling = GaussScale(2)
    scaling.setMu(np.asarray([[muX,muY]]).T)
    scaling.setSigma(np.asarray([sigmaX,sigmaY])) is used when Quadratic fit fails.'''
    pointX = np.asarray([x1,x2])
    
    y1 = mesh[:,0]
    y2 = mesh[:,1]
    
    xDrift = drift(pointX)
    f1x = xDrift[0][0]
    f2x = xDrift[0][1]
    yDrift = drift(mesh)
    f1y = yDrift[:,0]
    f2y = yDrift[:,1]
    
    xDiff = diff(pointX)
    g1x = xDiff[0,0]
    g2x = xDiff[1,1]
    yDiff = diff(mesh)
    g1y = yDiff[0,0]
    g2y = yDiff[1,1]
    
    
    # scale = h*g1(x1,x2)*g2(x1,x2)/(h*g1(y1,y2)*g2(y1,y2))
    # val = scale*np.exp(-(h**2*f1(y1,y2)**2+2*h*f1(y1,y2)*(x1-y1))/(2*h*g1(x1,x2)**2) + -(h**2*f2(y1,y2)**2+2*h*f2(y1,y2)*(x2-y2))/(2*h*g2(x1,x2)**2))*np.exp((x1-y1+h*f1(y1,y2))**2/(2*h*g1(x1,x2)**2) - (x1-y1+h *f1(y1,y2))**2/(2*h*g1(y1,y2)**2) + (x2-y2+h*f2(y1,y2))**2/(2*h*g2(x1,x2)**2) - (x2-y2+h* f2(y1,y2))**2/(2*h*g2(y1,y2)**2))
    # val = scale*np.exp(-(h**2*f1(y1,y2)**2-2*h*f1(y1,y2)*(x1-y1))/(2*h*g1(x1,x2)**2) + -(h**2*f2(y1,y2)**2-2*h*f2(y1,y2)*(x2-y2))/(2*h*g2(x1,x2)**2))*np.exp((x1-y1-h*f1(y1,y2))**2/(2*h*g1(x1,x2)**2) - (x1-y1-h *f1(y1,y2))**2/(2*h*g1(y1,y2)**2) + (x2-y2-h*f2(y1,y2))**2/(2*h*g2(x1,x2)**2) - (x2-y2-h* f2(y1,y2))**2/(2*h*g2(y1,y2)**2))
    
    scale = h*g1x*g2x/(h*g1y*g2y)
    val = scale*np.exp(-(h**2*f1y**2-2*h*f1y*(x1-y1))/(2*h*g1x**2) + -(h**2*f2y**2-2*h*f2y*(x2-y2))/(2*h*g2x**2))*np.exp((x1-y1-h*f1y)**2/(2*h*g1x**2) - (x1-y1-h *f1y)**2/(2*h*g1y**2) + (x2-y2-h*f2y)**2/(2*h*g2x**2) - (x2-y2-h* f2y)**2/(2*h*g2y**2))
    # assert np.isclose(scaleTest, scale).all()
    # assert np.isclose(val, valTest).all()
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
lejaPointsFinal, new = getLejaPoints(10, np.asarray([[0,0]]).T, poly, num_candidate_samples=5000, candidateSampleMesh = [], returnIndices = False)
    
def Test_LejaQuadratureLinearizationOnLejaPoints(mesh, pdf, poly, h, NumLejas, step, GMat, LPMat, LPMatBool,QuadFitMat,QuadFitBool, numQuadFit, twiceQuadFit):
    numLejas = LPMat.shape[1]
    # sigmaX=np.sqrt(h)*diff(np.asarray([[0,0]]))[0,0]
    # sigmaY=np.sqrt(h)*diff(np.asarray([[0,0]]))[1,1]
    
    newPDF = []
    condNums = []
    countUseMorePoints = 0 # Used to count if we have to revert to alternative procedure
    meshSize = len(mesh)
    LPUse = 0
    '''Try to Divide out Guassian using quadratic fit'''
    for ii in range(len(mesh)):
        # print('########################',ii/len(mesh)*100, '%')
        dr = h*drift(mesh[ii,:])
        muX = mesh[ii,0] #+ dr[0][0]
        muY = mesh[ii,1] #+ dr[0][1]
        
        scaling = GaussScale(2)
        scaling.setMu(np.asarray([[muX,muY]]).T)

        
        GPDF = np.expand_dims(GMat[ii,:meshSize], 1)*pdf
        # GPDF2 = np.expand_dims(GVals2(muX, muY, mesh, h),1)*pdf
        # assert np.max(abs(GPDF2-GPDF)) < 10**(-7)
        
        value, condNum, scaleUsed, LPMat, LPMatBool,QuadFitMat, QuadFitBool, reuseLP = QuadratureByInterpolationND_DivideOutGaussian(scaling, h, poly, mesh, GPDF, LPMat, LPMatBool,ii,NumLejas,QuadFitMat,QuadFitBool, numQuadFit, twiceQuadFit)
        LPUse = LPUse+reuseLP
        '''Alternative Method'''
        if math.isnan(condNum) or value <0 or condNum >10: 
            scaling.setCov((h*diff(np.asarray([muX,muY]))*diff(np.asarray([muX,muY])).T).T)
            mesh12 = VT.map_from_canonical_space(lejaPointsFinal, scaling)
            meshLP, distances, indx = UM.findNearestKPoints(scaling.mu[0][0],scaling.mu[1][0], mesh,numQuadFit, getIndices = True)
            pdfNew = pdf[indx]
            
            pdf12 = np.asarray(griddata(meshLP, pdfNew, mesh12, method='cubic', fill_value=0))
            pdfNew[pdfNew < 0] = 10**(-8)
            
            v = np.expand_dims(G(0,mesh12, h),1)
            
            integrand1 = newIntegrand(muX, muY, mesh12, h)
            testing1 = np.squeeze(pdf12)*integrand1
            # testing1 = np.squeeze(pdf12)*testing1
            # testing = np.squeeze(pdfNew)*integrand
            
            g = Gaussian(scaling, mesh12)
            testing = np.squeeze((pdf12*v)/np.expand_dims(g,1))
            # testing = np.squeeze(pdfNew)*integrand
            
            value, condNum = QuadratureByInterpolation_Simple(poly, scaling, mesh12, testing)
            value = value*(1/np.sqrt(2))
            countUseMorePoints = countUseMorePoints+1
            
            if value <0 :
                value = 10**(-8)

        newPDF.append(value)
        condNums.append(condNum)
    print('\n',(countUseMorePoints/len(mesh))*100, "% Used Interpolation**************")
    print('\n',(LPUse/len(mesh))*100, "% Reused Leja Points")

    newPDFs = np.asarray(newPDF)
    condNums = np.asarray([condNums]).T
    
    
    return newPDFs,condNums, mesh, LPMat, LPMatBool,QuadFitMat,QuadFitBool, LPUse, countUseMorePoints


