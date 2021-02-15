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

  
def QuadratureByInterpolationND(poly, scaling, mesh, pdf, NumLejas):
    '''Quadrature rule with change of variables for nonzero covariance. 
    Used by QuadratureByInterpolationND_DivideOutGaussian
    Selects a Leja points subset of the passed in mesh'''
    u = VT.map_to_canonical_space(mesh, scaling)
  
    normScale = GaussScale(2)
    normScale.setMu(np.asarray([[0,0]]).T)
    normScale.setCov(np.asarray([[1,0],[0,1]]))
    
    mesh2, pdfNew, indices = LP.getLejaSetFromPoints(normScale, u, NumLejas, poly, pdf)
    if math.isnan(indices[0]):
        return [10000], 10000, 10000
    assert np.max(indices) < len(mesh)

    numSamples = len(mesh2)          
    V = opolynd.opolynd_eval(mesh2, poly.lambdas[:numSamples,:], poly.ab, poly)
    try:
        vinv = np.linalg.inv(V)
    except np.linalg.LinAlgError as err: 
        if 'Singular matrix' in str(err):
            return [1000], 1000, indices
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
    pdfNew = pdf[LejaIndices]

    numSamples = len(mesh2)          
    V = opolynd.opolynd_eval(mesh2, poly.lambdas[:numSamples,:], poly.ab, poly)
    try:
        vinv = np.linalg.inv(V)
    except np.linalg.LinAlgError as err: 
        if 'Singular matrix' in str(err):
        # print("Singular******************")
            return 100000, 100000
    c = np.matmul(vinv, pdfNew)
    L = np.linalg.cholesky((scaling.cov))
    JacFactor = np.prod(np.diag(L))
    
    return c[0]*JacFactor*np.pi, np.sum(np.abs(vinv[0,:]))



def QuadratureByInterpolationND_DivideOutGaussian(scaling, h, poly, fullMesh, fullPDF, LPMat, LPMatBool, index, NumLejas, numQuadPoints):
    '''Divides out Gaussian using a quadratic fit. Then computes the update using a Leja Quadrature rule.'''
    x,y = fullMesh.T
    if not LPMatBool[index][0]: # Do not have points for quadratic fit
        mesh, distances, ii = UM.findNearestKPoints(scaling.mu[0][0],scaling.mu[1][0], fullMesh,numQuadPoints, getIndices = True)
        mesh =  mesh[:numQuadPoints]
        pdf = fullPDF[ii[:numQuadPoints]]
        scale1, temp, cc, Const = fitQuad(mesh, pdf)
        
    else:
        QuadPoints = LPMat[index,:].astype(int)
        mesh = fullMesh[QuadPoints]
        # plt.figure()
        # plt.scatter(mesh[:,0], mesh[:,1])
        # plt.scatter(fullMesh[index,0], fullMesh[index,1])
        pdf = fullPDF[QuadPoints]
        scale1, temp, cc, Const = fitQuad(mesh, pdf)
        
        
    if not math.isnan(Const): # succeeded fitting Gaussian
        x,y = fullMesh.T
        vals = np.exp(-(cc[0]*x**2+ cc[1]*y**2 + 2*cc[2]*x*y + cc[3]*x + cc[4]*y + cc[5]))/Const
        pdf2 = fullPDF/vals.T
        if LPMatBool[index][0]: # Don't Need LejaPoints
            LejaIndices = LPMat[index,:].astype(int)
            value, condNum = QuadratureByInterpolationND_KnownLP(poly, scale1, fullMesh, pdf2, LejaIndices)
            if condNum > 1.1:
                LPMatBool[index]=False
            else:
                return value[0], condNum, scale1, LPMat, LPMatBool, 1
            
        if not LPMatBool[index][0]: # Need Leja points.
            value, condNum, indices = QuadratureByInterpolationND(poly, scale1, fullMesh, pdf2,NumLejas)
            LPMat[index, :] = np.asarray(indices)
            if condNum < 1.1:
                LPMatBool[index] = True
            else: 
                LPMatBool[index] = False
            return value[0], condNum, scale1, LPMat, LPMatBool,0
    return float('nan'), float('nan'), float('nan'), LPMat, LPMatBool,0

    # elif LPMatBool[index][0]:
    #     value, condNum = QuadratureByInterpolationND_KnownLP(poly, scaling, mesh, pdf, LejaIndices)
        
        
        
        
        
        
        
