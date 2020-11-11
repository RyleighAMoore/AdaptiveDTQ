"""
Created on Fri Apr  3 12:44:33 2020
@author: Rylei
"""
from pyopoly1 import variableTransformations as VT
import numpy as np
import matplotlib.pyplot as plt
from pyopoly1 import opolynd
from mpl_toolkits.mplot3d import Axes3D
from Functions import *
from pyopoly1.Scaling import GaussScale
from pyopoly1.Plotting import productGaussians2D
import UnorderedMesh as UM
from pyopoly1.families import HermitePolynomials
import pyopoly1.indexing
import pyopoly1.LejaPoints as LP
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
    '''Quadrature rule with no change of variables. Must pass in mesh you want to use.
    Only works with Gaussian that has 0 covariance.'''
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
    '''Quadrature rule with change of variables for nonzero covariance. 
    Used by QuadratureByInterpolationND_DivideOutGaussian
    Selects a Leja points subset of the passed in mesh'''
    u = VT.map_to_canonical_space(mesh, scaling)
  
    normScale = GaussScale(2)
    normScale.setMu(np.asarray([[0,0]]).T)
    normScale.setCov(np.asarray([[1,0],[0,1]]))
    
    mesh2, pdfNew, indices = LP.getLejaSetFromPoints(normScale, u, 12, poly, pdf)

    numSamples = len(mesh2)          
    V = opolynd.opolynd_eval(mesh2, poly.lambdas[:numSamples,:], poly.ab, poly)
    vinv = np.linalg.inv(V)
    c = np.matmul(vinv, pdfNew)
    L = np.linalg.cholesky((scaling.cov))
    JacFactor = np.prod(np.diag(L))
    
    return c[0]*JacFactor*np.pi, np.sum(np.abs(vinv[0,:])), indices


def QuadratureByInterpolationND_KnownLP(poly, scaling, mesh, pdf, LejaIndices):
    '''Quadrature rule with change of variables for nonzero covariance. 
    Used by QuadratureByInterpolationND_DivideOutGaussian
    Selects a Leja points subset of the passed in mesh'''
    LejaMesh = mesh[LejaIndices]
    mesh2 = VT.map_to_canonical_space(LejaMesh, scaling)
    # mesh2 = LejaMeshCanonical
    pdfNew = pdf[LejaIndices]

    numSamples = len(mesh2)          
    V = opolynd.opolynd_eval(mesh2, poly.lambdas[:numSamples,:], poly.ab, poly)
    vinv = np.linalg.inv(V)
    c = np.matmul(vinv, pdfNew)
    L = np.linalg.cholesky((scaling.cov))
    JacFactor = np.prod(np.diag(L))
    
    return c[0]*JacFactor*np.pi, np.sum(np.abs(vinv[0,:]))



def QuadratureByInterpolationND_DivideOutGaussian(scaling, h, poly, fullMesh, fullPDF, LPMat, LPMatBool, index):
    '''Divides out Gaussian using a quadratic fit. Then computes the update using a Leja Quadrature rule.'''
    # if not LPMatBool[index][0]:
    x,y = fullMesh.T

    mesh, distances, ii = UM.findNearestKPoints(scaling.mu[0][0],scaling.mu[1][0], fullMesh, 20, getIndices = True)
    pdf = fullPDF[ii]
    
    scale1, temp, cc, Const = fitQuad(scaling.mu[0][0],scaling.mu[1][0], mesh, pdf)
    if not math.isnan(Const): # succeeded fitting Gaussian
        x,y = fullMesh.T
        vals = np.exp(-(cc[0]*x**2+ cc[1]*y**2 + 2*cc[2]*x*y + cc[3]*x + cc[4]*y + cc[5]))/Const
        pdf2 = fullPDF/vals.T
        if LPMatBool[index][0]: # Don't Need LejaPoints
            LejaIndices = LPMat[index,:].astype(int)
            value, condNum = QuadratureByInterpolationND_KnownLP(poly, scale1, fullMesh, pdf2, LejaIndices)
            if condNum > 3:
                LPMatBool[index]=False
            else:
                return value[0], condNum, scale1, LPMat, LPMatBool


        if not LPMatBool[index][0]: # Need Leja points.
            value, condNum, indices = QuadratureByInterpolationND(poly, scale1, fullMesh, pdf2)
            LPMat[index, :] = np.asarray(indices)
            LPMatBool[index] = True
            return value[0], condNum, scale1, LPMat, LPMatBool
    return float('nan'), float('nan'), float('nan'), LPMat, LPMatBool

    # elif LPMatBool[index][0]:
    #     value, condNum = QuadratureByInterpolationND_KnownLP(poly, scaling, mesh, pdf, LejaIndices)
        
        
        
        
        
        
        
