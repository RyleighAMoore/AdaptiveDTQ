# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 15:14:41 2020

@author: Rylei
"""
import variableTransformations as VT
import numpy as np
import matplotlib.pyplot as plt
import opolynd
from mpl_toolkits.mplot3d import Axes3D
from families import HermitePolynomials
import indexing
import sys
import QuadratureRules as QR
sys.path.insert(1, r'C:\Users\Rylei\Documents\SimpleDTQ')
from Functions import *
import UnorderedMesh as UM
from LejaPoints import getLejaPoints,mapPointsBack


# def productGaussians2D(muX1, muY1, muX2, muY2, sX1, sY1, sX2, sY2):
#     cov1 = np.zeros((2,2)); cov1[0,0] = sX1**2; cov1[1,1] = sY1**2
#     cov2 = np.zeros((2,2)); cov2[0,0] = sX2**2; cov2[1,1] = sY2**2
#     mu1 = np.zeros((2,1)); mu1[0] =muX1; mu1[1] = muY1  
#     mu2 = np.zeros((2,1)); mu2[0] =muX2; mu1[1] = muY2 
    
#     sigmaNew = np.linalg.inv(np.linalg.inv(cov1)+ np.linalg.inv(cov2))     
#     muNew = np.matmul(np.linalg.inv(np.linalg.inv(cov1) + np.linalg.inv(cov2)), np.matmul(np.linalg.inv(cov1),mu1) + np.matmul(np.linalg.inv(cov2),mu2))
    
#     c = 1/(np.sqrt(np.linalg.det(2*np.pi*(cov1+cov2))))
#     cc = np.matmul(np.matmul(-(1/2)*(mu1-mu2).T, np.linalg.inv(cov1+cov2)),(mu1-mu2))
#     cfinal = c*np.exp(cc)
#     return muNew, np.sqrt(sigmaNew), cfinal[0][0]

from Scaling import GaussScale

def productGaussians2D(scaling,scaling2):
    
    sigmaNew = np.linalg.inv(np.linalg.inv(scaling.cov)+ np.linalg.inv(scaling2.cov))     
    muNew = np.matmul(np.linalg.inv(np.linalg.inv(scaling.cov) + np.linalg.inv(scaling2.cov)), np.matmul(np.linalg.inv(scaling.cov),scaling.mu) + np.matmul(np.linalg.inv(scaling2.cov),scaling2.mu))
    
    c = 1/(np.sqrt(np.linalg.det(2*np.pi*(scaling.cov+scaling2.cov))))
    cc = np.matmul(np.matmul(-(1/2)*(scaling.mu-scaling2.mu).T, np.linalg.inv(scaling.cov+scaling2.cov)),(scaling.mu-scaling2.mu))
    cfinal = c*np.exp(cc)
    
    scale = GaussScale(len(muNew))
    scale.setMu(muNew)
    scale.setCov(sigmaNew)
    
    return scale, cfinal[0][0]

# from Scaling import GaussScale
# mu, cov, cfinal= productGaussians2D(1, 1, 0, 0, 0.1, 0.1, 0.5, 0.5)

# scale = GaussScale(2)
# scale.setMu(np.asarray([[0,0]]).T)
# scale.setSigma(np.asarray([0.5,0.5]))

# scale2 = GaussScale(2)
# scale2.setMu(np.asarray([[1,1]]).T)
# scale2.setSigma(np.asarray([0.1,0.1]))

# mu2, cov2, cfinal2= productGaussians2D2(scale, scale2)


def PlotG(Px, Py, h):
    mesh = UM.generateOrderedGridCenteredAtZero(-0.5, 0.5, -0.5, 0.5, 0.05, includeOrigin=True)
    G = GVals(Px,Py,mesh, h)
    # N = Gaussian(Px, Py, np.sqrt(h)*g1(), np.sqrt(h)*g2(), mesh)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(mesh[:,0], mesh[:,1], G,  c='k', marker='o')
    # ax.scatter(mesh[:,0], mesh[:,1], N,  c='r', marker='.')
    plt.show()
    
# PlotG(0,0,0.01)

def PlotH(Px, Py, h):
    mesh = UM.generateOrderedGridCenteredAtZero(-0.5, 0.5, -0.5, 0.5, 0.05, includeOrigin=True)
    H = HVals(Px,Py,mesh, h)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(mesh[:,0], mesh[:,1], H,  c='k', marker='o')
    plt.show()
    
# PlotH(0,0,0.01)

def PlotGH(Px, Py, h):
    mesh = UM.generateOrderedGridCenteredAtZero(-0.5, 0.5, -0.5, 0.5, 0.05, includeOrigin=True)
    H = HVals(Px,Py,mesh, h)
    G = GVals(Px,Py,mesh, h)
    Normal = Gaussian(Px, Py, np.sqrt(h)*g1(), np.sqrt(h)*g2(), mesh)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(mesh[:,0], mesh[:,1], H*Normal,  c='k', marker='o')
    ax.scatter(mesh[:,0], mesh[:,1], G,  c='r', marker='.')
    plt.show()

# PlotGH(0,0,0.01)






# mesh = UM.generateOrderedGridCenteredAtZero(-0.3, 0.3, -0.3, 0.3, 0.01, includeOrigin=True)
# H = HermitePolynomials(rho=0)
# d=2
# k = 20    
# ab = H.recurrence(k+1)
# lambdas = indexing.total_degree_indices(d, k)
# H.lambdas = lambdas

# mesh, two = getLejaPoints(230, np.asarray([[0,0]]).T,H,candidateSampleMesh = [], returnIndices = False)
# plt.scatter(mesh[:,0], mesh[:,1])
# mesh = mapPointsBack(0, 0, mesh, 0.1, 0.1)

# pdf = Gaussian(0, 0,0.1,0.1,mesh)*GVals(0, 0, mesh, 0.01)
# # muNew, sigmaNew, cfinal = productGaussians2D(0, 0, 0, 0, 0.1, 0.1, 0.1, 0.1)
# scaling, newPDF = GetGaussianPart(0, 0, mesh, pdf, 0.01)

# value, condNum = QR.QuadratureByInterpolationND(H, scaling, mesh, newPDF)
# print(value,condNum)