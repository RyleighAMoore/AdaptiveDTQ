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
import Functions as fun

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


# def QuadratureByInterpolation1D(poly, scaling, mesh, pdf):
#     xCan=VT.map_to_canonical_space(mesh, scaling) 
#     V = poly.eval(xCan, range(len(xCan)))
#     vinv = np.linalg.inv(V)
#     c = np.matmul(vinv, pdf)
#     plot=False
#     if plot:
#         interp = np.matmul(V,c)
#         plt.figure()
#         plt.plot(mesh, interp,'.')
#         plt.plot(mesh, pdf)
#     return c[0]


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

  
def QuadratureByInterpolationND(poly, scaling, mesh, pdf, LejaIndices):
    '''Quadrature rule with change of variables for nonzero covariance. 
    Used by QuadratureByInterpolationND_DivideOutGaussian
    Selects a Leja points subset of the passed in mesh'''
    if np.min(LejaIndices)<0: # Need to compute Leja points
        u = VT.map_to_canonical_space(mesh, scaling)
        normScale = GaussScale(2)
        normScale.setMu(np.asarray([[0,0]]).T)
        normScale.setCov(np.asarray([[1,0],[0,1]]))
        mesh2, pdfNew, LejaIndices = LP.getLejaSetFromPoints(normScale, u, 12, poly, pdf)
        # weight = fun.Gaussian(scaling, mesh)
        
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # ax.scatter(u[:,0], u[:,1], weight, c='k', marker='o')
        # ax.scatter(mesh2[:,0], mesh2[:,1], np.max(weight)+1, c='b', marker='o')
        # # ax.scatter(mesh2[:,0], mesh2[:,1], np.max(weight)+1, c='r', marker='.')
        # plt.show()
    else: 
        LejaMesh = mesh[LejaIndices]
        mesh2 = VT.map_to_canonical_space(LejaMesh, scaling)
        # mesh2 = LejaMeshCanonical
        pdfNew = pdf[LejaIndices]
        
        # mesh11 = VT.map_to_canonical_space(mesh, scaling)
        
        # u = VT.map_to_canonical_space(mesh, scaling)
        # normScale = GaussScale(2)
        # normScale.setMu(np.asarray([[0,0]]).T)
        # normScale.setCov(np.asarray([[1,0],[0,1]]))
        # mesh21, pdf21, indices21 = LP.getLejaSetFromPoints(normScale, u, 20, poly, pdf)
        
        # weight = fun.Gaussian(scaling, mesh)
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # ax.scatter(u[:,0], u[:,1], weight, c='k', marker='o')
        # ax.scatter(mesh21[:,0], mesh21[:,1], np.max(weight)+1, c='b', marker='o')
        # ax.scatter(mesh2[:,0], mesh2[:,1], np.max(weight)+2, c='r', marker='o')
        
        # plt.show()
        
        # numSamples = len(mesh2)          
        # V = opolynd.opolynd_eval(mesh2, poly.lambdas[:numSamples,:], poly.ab, poly)
        # vinv = np.linalg.inv(V)
        # c = np.matmul(vinv, pdfNew)
        # L = np.linalg.cholesky((scaling.cov))
        # JacFactor = np.prod(np.diag(L))
        # sol1 = c[0]*JacFactor*np.pi
        
        # ###
        # u = VT.map_to_canonical_space(mesh, scaling)
        # normScale = GaussScale(2)
        # normScale.setMu(np.asarray([[0,0]]).T)
        # normScale.setCov(np.asarray([[1,0],[0,1]]))
        # mesh2, pdfNew, indices = LP.getLejaSetFromPoints(normScale, u, 20, poly, pdf)
        # numSamples = len(mesh2)          
        # V = opolynd.opolynd_eval(mesh2, poly.lambdas[:numSamples,:], poly.ab, poly)
        # vinv = np.linalg.inv(V)
        # c = np.matmul(vinv, pdfNew)
        # L = np.linalg.cholesky((scaling.cov))
        # JacFactor = np.prod(np.diag(L))
        # sol2 = c[0]*JacFactor*np.pi
        # ###
        # # print(abs(sol1-sol2))
        # if abs(sol1-sol2) > 0.08:
        #     p=0

    numSamples = len(mesh2)          
    V = opolynd.opolynd_eval(mesh2, poly.lambdas[:numSamples,:], poly.ab, poly)
    try:
        vinv = np.linalg.inv(V)
        c = np.matmul(vinv, pdfNew)
        L = np.linalg.cholesky((scaling.cov))
        JacFactor = np.prod(np.diag(L))
    except:
        u = VT.map_to_canonical_space(mesh, scaling)
        normScale = GaussScale(2)
        normScale.setMu(np.asarray([[0,0]]).T)
        normScale.setCov(np.asarray([[1,0],[0,1]]))
        mesh2, pdfNew, LejaIndices = LP.getLejaSetFromPoints(normScale, u, 12, poly, pdf)
        numSamples = len(mesh2)          
        V = opolynd.opolynd_eval(mesh2, poly.lambdas[:numSamples,:], poly.ab, poly)
        vinv = np.linalg.inv(V)
        c = np.matmul(vinv, pdfNew)
        L = np.linalg.cholesky((scaling.cov))
        JacFactor = np.prod(np.diag(L))
    if not np.min(LejaIndices)<0 and np.sum(np.abs(vinv[0,:])) > 2: # Try to compute new LejaPoints
        # print('once')
        u = VT.map_to_canonical_space(mesh, scaling)
        normScale = GaussScale(2)
        normScale.setMu(np.asarray([[0,0]]).T)
        normScale.setCov(np.asarray([[1,0],[0,1]]))
        mesh2, pdfNew, LejaIndices = LP.getLejaSetFromPoints(normScale, u, 12, poly, pdf)
        numSamples = len(mesh2)          
        V = opolynd.opolynd_eval(mesh2, poly.lambdas[:numSamples,:], poly.ab, poly)
        vinv = np.linalg.inv(V)
        c = np.matmul(vinv, pdfNew)
        L = np.linalg.cholesky((scaling.cov))
        JacFactor = np.prod(np.diag(L))
    
    return c[0]*JacFactor*np.pi, np.sum(np.abs(vinv[0,:])), LejaIndices



def QuadratureByInterpolationND_DivideOutGaussian(scaling, h, poly, fullMesh, fullPDF, LejaIndices):
    '''Divides out Gaussian using a quadratic fit. Then computes the update using a Leja Quadrature rule.'''
    x,y = fullMesh.T
    if not np.min(LejaIndices)<0:
        mesh = fullMesh[LejaIndices.astype(int)]
        pdf = fullPDF[LejaIndices.astype(int)]
    else:
        mesh, distances, indices1 = UM.findNearestKPoints(scaling.mu[0][0],scaling.mu[1][0], fullMesh, 20, getIndices = True)
        pdf = fullPDF[indices1]
    
    
    if math.isnan(pdf[0]): # Failed getting leja points
        Const = float('nan')
        return float('nan'),float('nan'), float('nan'), 0

    else:
        scale1, temp, cc, Const = fitQuad(mesh, pdf)
        # weight = fun.Gaussian(scale1, fullMesh)
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # ax.scatter(fullMesh[:,0], fullMesh[:,1], weight, c='k', marker='o')
        # # ax.scatter(fullMesh[:,0], fullMesh[:,1], fullPDF, c='b', marker='o')
        # plt.show()
        
        
        if not math.isnan(Const): # succeeded fitting Gaussian
            x,y = fullMesh.T
            vals = np.exp(-(cc[0]*x**2+ cc[1]*y**2 + 2*cc[2]*x*y + cc[3]*x + cc[4]*y + cc[5]))/Const
            if np.min(vals)==0:
                return float('nan'),float('nan'), float('nan'), 0
            pdf2 = fullPDF/vals.T
            # print(np.min(vals))
            
        
            if np.min(LejaIndices)<0:
                value, condNum, indices = QuadratureByInterpolationND(poly, scale1, fullMesh, pdf2,LejaIndices)
            else:
                # LejaMeshCanonical=mesh
                # LejaPointPDFVals = pdf2[indices21]
                LejaIndices = LejaIndices.astype(int)
                LejaPointPDFVals = pdf2[np.ndarray.tolist(LejaIndices)]
                value, condNum, indices = QuadratureByInterpolationND(poly, scale1, fullMesh, pdf2, LejaIndices)
                
            if value > 50:
                return float('nan'),float('nan'), float('nan'), 0
            
            
            return value[0], condNum, scale1, indices       
    return float('nan'),float('nan'), float('nan'), 0


# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(x, y, pdf2, c='r', marker='.')
# plt.show()