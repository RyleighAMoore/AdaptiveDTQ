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


# def QuadratureByInterpolation_Simple(poly, scaling, mesh, pdf):
#     '''Quadrature rule with no change of variables. Must pass in mesh you want to use.
#     Only works with Gaussian that has 0 covariance.'''
#     u = VT.map_to_canonical_space(mesh, scaling)
#     normScale = GaussScale(2)
#     normScale.setMu(np.asarray([[0,0]]).T)
#     normScale.setCov(np.asarray([[1,0],[0,1]]))
    
#     mesh2 = u
#     pdfNew = pdf
    
#     numSamples = len(mesh2)          
#     V = opolynd.opolynd_eval(mesh2, poly.lambdas[:numSamples,:], poly.ab, poly)
#     vinv = np.linalg.inv(V)
#     c = np.matmul(vinv, pdfNew)
    
#     return c[0], np.sum(np.abs(vinv[0,:]))

  
def QuadratureByInterpolationND(poly, scaling, mesh, pdf, LejaMeshCanonical, LejaPointPDFVals, time=False):
    '''Quadrature rule with change of variables for nonzero covariance. 
    Used by QuadratureByInterpolationND_DivideOutGaussian
    Selects a Leja points subset of the passed in mesh'''
    if time:
        u = VT.map_to_canonical_space(mesh, scaling)
        normScale = GaussScale(2)
        normScale.setMu(np.asarray([[0,0]]).T)
        normScale.setCov(np.asarray([[1,0],[0,1]]))
        mesh2, pdfNew, indices = LP.getLejaSetFromPoints(normScale, u, 20, poly, pdf)
        # weight = fun.Gaussian(scaling, u)
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # ax.scatter(u[:,0], u[:,1], weight, c='k', marker='o')
        # # ax.scatter(mesh21[:,0], mesh21[:,1], np.max(weight)+1, c='b', marker='o')

        # # ax.scatter(mesh2[:,0], mesh2[:,1], np.max(weight)+1, c='r', marker='.')
        
        # plt.show()
        
    else:
        mesh2 = VT.map_to_canonical_space(LejaMeshCanonical, scaling)
        # mesh2 = LejaMeshCanonical
        pdfNew = LejaPointPDFVals
        indices = []
    
        # mesh11 = VT.map_to_canonical_space(mesh, scaling)
        
        u = VT.map_to_canonical_space(mesh, scaling)
        normScale = GaussScale(2)
        normScale.setMu(np.asarray([[0,0]]).T)
        normScale.setCov(np.asarray([[1,0],[0,1]]))
        mesh21, pdf21, indices21 = LP.getLejaSetFromPoints(normScale, u, 20, poly, pdf)
        
        weight = fun.Gaussian(scaling, mesh)
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # ax.scatter(u[:,0], u[:,1], weight, c='k', marker='o')
        # ax.scatter(mesh21[:,0], mesh21[:,1], np.max(weight)+1, c='b', marker='o')

        # ax.scatter(mesh2[:,0], mesh2[:,1], np.max(weight)+2, c='r', marker='o')
        
        # plt.show()
        
        numSamples = len(mesh2)          
        V = opolynd.opolynd_eval(mesh2, poly.lambdas[:numSamples,:], poly.ab, poly)
        vinv = np.linalg.inv(V)
        c = np.matmul(vinv, pdfNew)
        L = np.linalg.cholesky((scaling.cov))
        JacFactor = np.prod(np.diag(L))
        sol1 = c[0]*JacFactor*np.pi
        
        ###
        u = VT.map_to_canonical_space(mesh, scaling)
        normScale = GaussScale(2)
        normScale.setMu(np.asarray([[0,0]]).T)
        normScale.setCov(np.asarray([[1,0],[0,1]]))
        mesh2, pdfNew, indices = LP.getLejaSetFromPoints(normScale, u, 20, poly, pdf)
        numSamples = len(mesh2)          
        V = opolynd.opolynd_eval(mesh2, poly.lambdas[:numSamples,:], poly.ab, poly)
        vinv = np.linalg.inv(V)
        c = np.matmul(vinv, pdfNew)
        L = np.linalg.cholesky((scaling.cov))
        JacFactor = np.prod(np.diag(L))
        sol2 = c[0]*JacFactor*np.pi
        ###
        # print(abs(sol1-sol2))
        if abs(sol1-sol2) > 0.08:
            p=0

    numSamples = len(mesh2)          
    V = opolynd.opolynd_eval(mesh2, poly.lambdas[:numSamples,:], poly.ab, poly)
    vinv = np.linalg.inv(V)
    c = np.matmul(vinv, pdfNew)
    L = np.linalg.cholesky((scaling.cov))
    JacFactor = np.prod(np.diag(L))
    if  np.sum(np.abs(vinv[0,:])) > 5:
        ttt=0
    
    return c[0]*JacFactor*np.pi, np.sum(np.abs(vinv[0,:])), indices



def QuadratureByInterpolationND_DivideOutGaussian(scaling, h, poly, fullMesh, fullPDF, LejaMeshCanonical, LejaIndices, time=False):
    '''Divides out Gaussian using a quadratic fit. Then computes the update using a Leja Quadrature rule.'''
    x,y = fullMesh.T
    
    mesh, distances, indices1 = UM.findNearestKPoints(scaling.mu[0][0],scaling.mu[1][0], fullMesh, 20, getIndices = True)
    pdf = fullPDF[indices1]
    
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
        pdf2 = fullPDF/vals.T
        
        if time:
            value, condNum, indices = QuadratureByInterpolationND(poly, scale1, fullMesh, pdf2, [], [], time=True)

        else:
            # LejaMeshCanonical=mesh
            # LejaPointPDFVals = pdf2[indices21]
            LejaPointPDFVals = pdf2[np.ndarray.tolist(LejaIndices)]
            value, condNum, indices = QuadratureByInterpolationND(poly, scale1, fullMesh, pdf2, LejaMeshCanonical, LejaPointPDFVals)

        
        return value[0], condNum, scale1, indices       
    return float('nan'),float('nan'), float('nan'), 0


# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(x, y, pdf2, c='r', marker='.')
# plt.show()