"""
Created on Fri Apr  3 12:44:33 2020
@author: Rylei
"""
import variableTransformations as VT
import numpy as np
import matplotlib.pyplot as plt
import opolynd
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.insert(1, r'C:\Users\Rylei\Documents\SimpleDTQ')
from Functions import *
from Scaling import GaussScale
from Plotting import productGaussians2D
import UnorderedMesh as UM
from families import HermitePolynomials
import indexing
import LejaPoints as LP
from QuadraticFit import fitQuad
from scipy.interpolate import griddata
import math

def getValsWithinRadius(Px,Py,canonicalMesh, pdf, numCandidiates):
    point = np.asarray([Px,Py])
    normList =np.sqrt(np.sum((point*np.shape(canonicalMesh)-canonicalMesh)**2,axis=1))
    meshVals = []
    pdfVals = []
    for val in range(len(normList)):
        if normList[val] < np.sqrt(2)*numCandidiates:
           meshVals.append(canonicalMesh[val])
           pdfVals.append(pdf[val])
    return np.asarray(meshVals), np.asarray(pdfVals)


def QuadratureByInterpolation1D(poly, scaling, mesh, pdf):
    xCan=VT.map_to_canonical_space(mesh, scaling) 
    V = poly.eval(xCan, range(len(xCan)))
    vinv = np.linalg.inv(V)
    c = np.matmul(vinv, pdf)
    plot=False
    if plot:
        interp = np.matmul(V,c)
        plt.figure()
        plt.plot(mesh, interp,'.')
        plt.plot(mesh, pdf)
    return c[0]


def QuadratureByInterpolation_Simple(poly, scaling, mesh, pdf):
    u = VT.map_to_canonical_space(mesh, scaling)
    normScale = GaussScale(2)
    normScale.setMu(np.asarray([[0,0]]).T)
    normScale.setCov(np.asarray([[1,0],[0,1]]))
    
    mesh2 = u
    pdfNew = pdf
    
    numSamples = len(mesh2)          
    V = opolynd.opolynd_eval(mesh2, poly.lambdas[:numSamples,:], poly.ab, poly)
    vinv = np.linalg.inv(V)
    c = np.matmul(vinv, pdfNew)
    
    return c[0], np.sum(np.abs(vinv[0,:]))


  
def QuadratureByInterpolationND(poly, scaling, mesh, pdf):
    u = VT.map_to_canonical_space(mesh, scaling)
  
    normScale = GaussScale(2)
    normScale.setMu(np.asarray([[0,0]]).T)
    normScale.setCov(np.asarray([[1,0],[0,1]]))
    
    mesh2, pdfNew = LP.getLejaSetFromPoints(normScale, u, 12, poly, pdf)

    numSamples = len(mesh2)          
    V = opolynd.opolynd_eval(mesh2, poly.lambdas[:numSamples,:], poly.ab, poly)
    vinv = np.linalg.inv(V)
    c = np.matmul(vinv, pdfNew)
    L = np.linalg.cholesky((scaling.cov))
    JacFactor = np.prod(np.diag(L))
    
    return c[0]*JacFactor*np.pi, np.sum(np.abs(vinv[0,:]))



def QuadratureByInterpolationND_DivideOutGaussian(scaling, h, poly, fullMesh, fullPDF):
    Spread = 0.2
    x,y = fullMesh.T

    mesh, distances, indices = UM.findNearestKPoints(scaling.mu[0][0],scaling.mu[1][0], fullMesh, 20, getIndices = True)
    pdf = fullPDF[indices]
    
    
    value = float('nan')
    if math.isnan(pdf[0]): # Failed getting leja points
        Const = float('nan')
    else: # succeeded getting leja points
        scale1, temp, cc, Const = fitQuad(scaling.mu[0][0],scaling.mu[1][0], mesh, pdf)
        if not math.isnan(Const): # succeeded fitting Gaussian
            x,y = fullMesh.T
            vals = np.exp(-(cc[0]*x**2+ cc[1]*y**2 + 2*cc[2]*x*y + cc[3]*x + cc[4]*y + cc[5]))/Const
            pdf2 = fullPDF/vals.T
            value, condNum = QuadratureByInterpolationND(poly, scale1, fullMesh, pdf2)
            return value[0], condNum, scale1
            
    return float('nan'),float('nan'), float('nan')
