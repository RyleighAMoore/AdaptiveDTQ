
import numpy as np
from scipy.stats import uniform, beta, norm, multivariate_normal
from functools import partial
import numpy as np
import UnorderedMesh as UM
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Functions import *
from scipy.interpolate import griddata, interp2d 
from pyopoly1.LejaPoints import getLejaSetFromPoints, mapPointsBack, mapPointsTo, getLejaPoints
from pyopoly1.QuadratureRules import QuadratureByInterpolationND, QuadratureByInterpolationND_DivideOutGaussian
from pyopoly1.families import HermitePolynomials
from pyopoly1.Scaling import GaussScale
from pyopoly1 import indexing
import math
import Functions as fun


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
def Test_LejaQuadratureLinearizationOnLejaPoints(mesh, pdf, poly, h, numNodes, step, GMat, LPMatIndices, time=False):
    if time:
        LPMatIndices = np.empty([2000, 12])
    
    newPDF = []
    condNums = []
    countUseMorePoints = 0 # Used to count if we have to revert to alternative procedure
    
    Sigma = np.sqrt(h)*diff(mesh) # sigma of gaussian for weight
    '''Try to Divide out Guassian using quadratic fit'''
    for ii in range(len(mesh)):
    
        muX = mesh[ii,0] # mean of gaussian for weight
        muY = mesh[ii,1]
        # print(muX,muY)

        Gvalues = GMat[ii,:len(mesh)]
        
        scaling = GaussScale(2)
        scaling.setMu(np.asarray([[muX,muY]]).T)
        scaling.setSigma(np.diag(Sigma))
        
        GPDF = (Gvalues*pdf.T).T
        
        if time: 
            value, condNum, scaleUsed, indices = QuadratureByInterpolationND_DivideOutGaussian(scaling, h, poly, mesh, GPDF,[], time=True)
        else:
            value, condNum, scaleUsed, indices= QuadratureByInterpolationND_DivideOutGaussian(scaling, h, poly, mesh, GPDF, LPMatIndices[ii,:],time=False)

        if time and not math.isnan(condNum) or value < 0 or condNum > 10:
            LPs = []
            for i in range(len(indices)):
                LPs.append(mesh[indices[i],:])
                LejaPointIndices = indices
            aa = np.asarray(LPs)
            # mesh2 = VT.map_from_canonical_space(mesh2, scaling)
            LPMatIndices[ii,:len(indices)] = np.copy(indices)
        
        '''Alternative Method'''
        if math.isnan(condNum) or value < 0 or condNum > 10:
            LejaMeshCanonical=[]
            LejaPointPDFVals = []
            #Divide out by gaussian, pass into QuadratureByInterpolationND
            # value = 0.003
            scale = GaussScale(2)
            scale.setMu(np.asarray([[0,0]]).T)
            scale.setSigma(np.asarray([np.sqrt(h)*fun.diff(np.asarray([[0,0]]))[0,0],np.sqrt(h)*fun.diff(np.asarray([[0,0]]))[1,1]]))
            weight = fun.Gaussian(scale, mesh)
            GPDF2 = GPDF/np.expand_dims(weight, axis=1)
            value, condNum, indices = QuadratureByInterpolationND(poly, scale, mesh, GPDF2, LejaMeshCanonical, LejaPointPDFVals, time=True)           
            value =value[0]
            countUseMorePoints = countUseMorePoints+1
            if value <0:
                value = 10**(-8)

        newPDF.append(value)
        condNums.append(condNum)
        
    print('\n',(countUseMorePoints/len(mesh))*100, "% Used Interpolation*********************************")
    newPDFs = np.asarray(newPDF)
    condNums = np.asarray([condNums]).T
    if time:
        LPMatIndices = np.copy(LPMatIndices.astype(int))
    return newPDFs,condNums, mesh, LPMatIndices


