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


def productGaussians2D(muX1, muY1, muX2, muY2, sX1, sY1, sX2, sY2):
    cov1 = np.zeros((2,2)); cov1[0,0] = sX1**2; cov1[1,1] = sY1**2
    cov2 = np.zeros((2,2)); cov2[0,0] = sX2**2; cov2[1,1] = sY2**2
    mu1 = np.zeros((2,1)); mu1[0] =muX1; mu1[1] = muY1  
    mu2 = np.zeros((2,1)); mu2[0] =muX2; mu1[1] = muY2 
    
    sigmaNew = np.linalg.inv(np.linalg.inv(cov1)+ np.linalg.inv(cov2))     
    muNew = np.matmul(np.linalg.inv(np.linalg.inv(cov1) + np.linalg.inv(cov2)), np.matmul(np.linalg.inv(cov1),mu1) + np.matmul(np.linalg.inv(cov2),mu2))
    
    c = 1/(np.sqrt(np.linalg.det(2*np.pi*(cov1+cov2))))
    cc = np.matmul(np.matmul(-(1/2)*(mu1-mu2).T, np.linalg.inv(cov1+cov2)),(mu1-mu2))
    cfinal = c*np.exp(cc)
    return muNew, np.sqrt(sigmaNew), cfinal[0][0]

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

def GetGaussianPart(Px, Py, mesh, pdf, h):
    muX = np.mean(mesh[:,0]*np.sqrt(pdf))
    muY = np.mean(mesh[:,1]*np.sqrt(pdf))

    vals = np.cov(mesh.T, aweights =np.sqrt(pdf))
    sigmaX = np.sqrt(vals[0,0])
    sigmaY = np.sqrt(vals[1,1])
    muX = mesh[np.argmax(pdf),0]
    muY = mesh[np.argmax(pdf),1]
    sigmaX = np.round(sigmaX,1)
    sigmaY =np.round(sigmaY,1)
    Gauss = Gaussian(muX, muY, sigmaX, sigmaY, mesh)
    

    
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(mesh[:,0], mesh[:,1], np.log(Gauss),  c='k', marker='o')
    # ax.scatter(mesh[:,0], mesh[:,1], np.log(pdf/Gauss),  c='r', marker='.')
    # plt.show()
    
    
    fig = plt.figure()
    ax = Axes3D(fig)
    # ax.scatter(mesh[:,0], mesh[:,1], Gauss,  c='k', marker='o')
    ax.scatter(mesh[:,0], mesh[:,1], pdf/Gauss,  c='r', marker='.')
    plt.xlim([-5*sigmaX, 5*sigmaX])
    plt.ylim([-5*sigmaY, 5*sigmaY])
    plt.show()
    scaling = np.asarray([[muX, sigmaX], [muY, sigmaY]])
    
    return scaling, pdf/Gauss




mesh = UM.generateOrderedGridCenteredAtZero(-0.3, 0.3, -0.3, 0.3, 0.01, includeOrigin=True)
H = HermitePolynomials(rho=0)
d=2
k = 20    
ab = H.recurrence(k+1)
lambdas = indexing.total_degree_indices(d, k)
H.lambdas = lambdas

mesh, two = getLejaPoints(230, np.asarray([[0,0]]).T,H,candidateSampleMesh = [], returnIndices = False)
mesh = mapPointsBack(0, 0, mesh, 0.1, 0.1)

pdf = Gaussian(0, 0,0.1,0.1,mesh)*GVals(0, 0, mesh, 0.01)
# muNew, sigmaNew, cfinal = productGaussians2D(0, 0, 0, 0, 0.1, 0.1, 0.1, 0.1)
scaling, newPDF = GetGaussianPart(0, 0, mesh, pdf, 0.01)

value, condNum = QR.QuadratureByInterpolationND(H, scaling, mesh, newPDF)
print(value,condNum)