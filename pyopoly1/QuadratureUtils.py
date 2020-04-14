# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 15:45:10 2020

@author: Rylei
"""
from Scaling import GaussScale
import numpy as np
from Functions import Gaussian
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def GetGaussianPart(mesh, pdf, h, round=1):
    muX = np.mean(mesh[:,0]*pdf)
    muY = np.mean(mesh[:,1]*pdf)
    muX = mesh[np.argmax(pdf),0]
    muY = mesh[np.argmax(pdf),1]

    vals = np.cov(mesh.T, aweights = np.squeeze(pdf))
    # sigmaX = np.sqrt(vals[0,0])
    # sigmaY = np.sqrt(vals[1,1])

    # sigmaX = np.round(sigmaX,round)
    # sigmaY =np.round(sigmaY,round)
    
    scale = GaussScale(2)
    scale.setMu(np.asarray([[muX,muY]]).T)
    scale.cov =vals
    
    
    Gauss = Gaussian(scale, mesh)
    
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(mesh[:,0], mesh[:,1], pdf/np.expand_dims(Gauss,1),  c='k', marker='o')
    # # ax.scatter(mesh[:,0], mesh[:,1], np.log(pdf/Gauss),  c='r', marker='.')
    # plt.show()
    
    
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(mesh[:,0], mesh[:,1], Gauss,  c='k', marker='o')
    # ax.scatter(mesh[:,0], mesh[:,1], pdf/Gauss,  c='r', marker='.')
    # plt.xlim([-5*sigmaX, 5*sigmaX])
    # plt.ylim([-5*sigmaY, 5*sigmaY])
    # plt.show()
    # print(scale.cov)
    # print(scale.mu)
    return scale, pdf/np.expand_dims(Gauss,1)