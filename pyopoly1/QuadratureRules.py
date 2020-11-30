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

  
def QuadratureByInterpolationND(poly, scaling, mesh, pdf, NumLejas):
    '''Quadrature rule with change of variables for nonzero covariance. 
    Used by QuadratureByInterpolationND_DivideOutGaussian
    Selects a Leja points subset of the passed in mesh'''
    u = VT.map_to_canonical_space(mesh, scaling)
  
    normScale = GaussScale(2)
    normScale.setMu(np.asarray([[0,0]]).T)
    normScale.setCov(np.asarray([[1,0],[0,1]]))
    
    # u = u[NearestIndices.astype(int)]
    # # print(u.shape)
    # pdf = pdf[NearestIndices.astype(int)]
    # plt.figure()
    # plt.plot(u[:,0], u[:,1],'ok')
    # plt.plot(mesh[:,0], mesh[:,1],'.r')
    # # plt.plot(mesh[index,0], fullMesh[index,1], 'og')
    # plt.show()
    
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
            # plt.figure()
            # plt.plot(mesh[:,0], mesh[:,1], 'ok')
            # plt.plot(scaling.mu[0], scaling.mu[1], '.r')
            # plt.show()
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
    # mesh2 = LejaMeshCanonical
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



def QuadratureByInterpolationND_DivideOutGaussian(scaling, h, poly, fullMesh, fullPDF, LPMat, LPMatBool, index, NumLejas, QuadFitMat,QuadFitBool, numQuadPoints):
    '''Divides out Gaussian using a quadratic fit. Then computes the update using a Leja Quadrature rule.'''
    # if not LPMatBool[index][0]:
    x,y = fullMesh.T
    if not QuadFitBool[index]:
        mesh, distances, ii = UM.findNearestKPoints(scaling.mu[0][0],scaling.mu[1][0], fullMesh,numQuadPoints, getIndices = True)
        # plt.figure()
        # plt.plot(fullMesh[:,0], fullMesh[:,1],'ok')
        # plt.plot(mesh[:,0], mesh[:,1],'.r')
        # plt.plot(fullMesh[index,0], fullMesh[index,1], 'og')
        # plt.show()
        
        mesh =  mesh[:numQuadPoints]
        pdf = fullPDF[ii[:numQuadPoints]]
        scale1, temp, cc, Const = fitQuad(mesh, pdf)
        QuadFitMat[index,:] = ii
        
    else:
        mesh = fullMesh[LejaIndices]
        pdf = fullPDF[LejaIndices]
        scale1, temp, cc, Const = fitQuad(mesh, pdf)
        
    if not math.isnan(Const): # succeeded fitting Gaussian
        x,y = fullMesh.T
        vals = np.exp(-(cc[0]*x**2+ cc[1]*y**2 + 2*cc[2]*x*y + cc[3]*x + cc[4]*y + cc[5]))/Const
        pdf2 = fullPDF/vals.T
        if LPMatBool[index][0]: # Don't Need LejaPoints
            LejaIndices = LPMat[index,:].astype(int)
            value, condNum = QuadratureByInterpolationND_KnownLP(poly, scale1, fullMesh, pdf2, LejaIndices)
            if condNum > 1.05:
                LPMatBool[index]=False
                QuadFitBool[index] = False
            else:
                # print("LP Reused")
                return value[0], condNum, scale1, LPMat, LPMatBool, QuadFitMat,QuadFitBool, 1
            
        if not LPMatBool[index][0]: # Need Leja points.
            value, condNum, indices = QuadratureByInterpolationND(poly, scale1, fullMesh, pdf2,NumLejas)
            LPMat[index, :] = np.asarray(indices)
            if condNum < 10:
                LPMatBool[index] = True
            else: 
                LPMatBool[index] = False
            return value[0], condNum, scale1, LPMat, LPMatBool, QuadFitMat,QuadFitBool,0
    return float('nan'), float('nan'), float('nan'), LPMat, LPMatBool, QuadFitMat,QuadFitBool,0

    # elif LPMatBool[index][0]:
    #     value, condNum = QuadratureByInterpolationND_KnownLP(poly, scaling, mesh, pdf, LejaIndices)
        
        
        
        
        
        
        
