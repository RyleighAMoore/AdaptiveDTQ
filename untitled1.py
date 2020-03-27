# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 18:57:47 2020

@author: Rylei
"""
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import Functions as fun
import Integrand
import Operations2D
import XGrid
from mpl_toolkits.mplot3d import Axes3D
import QuadRules
from tqdm import tqdm, trange
import random
import UnorderedMesh as UM
from scipy.spatial import Delaunay
import MeshUpdates2D as MeshUp
import pickle
import os
import datetime
import time
import GenerateLejaPoints as LP
import pickle
import LejaQuadrature as LQ
import getPCE as PCE
import distanceMetrics as DM
import scipy as sp
from scipy.stats import multivariate_normal



polyLarge = PCE.generatePCE(40, muX=0, muY=0, sigmaX = 1, sigmaY=1)
poly = PCE.generatePCE(20, muX=0, muY=0, sigmaX = 1, sigmaY=1)

train_samples1, weights1 = LQ.getLejaQuadratureRule(1, 1 ,1,1)


D = len(poly.indices.T) - 1
Kmax = 530 # Will give Kmax+1 samples
samples, mesh2 = LP.getLejaPointsWithStartingPoints([0,0,1,1], Kmax, 5000, polyLarge)

initSamples = train_samples1
otherSamples = samples
otherSamples = np.ndarray.tolist(otherSamples)
otherSamples.pop()

vmat = poly.basis_matrix(samples.T).T

weights = weights1.T
print(np.sum(weights))
nodes = np.copy(initSamples)
for K in range(D, Kmax): # up to Kmax - 1
    # Add Node
    nodes = np.vstack((nodes, otherSamples.pop()))
    one = ((K+1)/(K+2))*weights
    two = np.asarray([[1/(K+2+1)]])

    weights = np.concatenate((one, two))
    
    # Update weights
    vmat = poly.basis_matrix(nodes.T).T
    nullspace = sp.linalg.null_space(vmat)



    c = np.asarray([nullspace[:,0]]).T
    
    a = weights/c
    aPos = np.ma.masked_where(c<0, a) # only values where c > 0 
    alphaMax = np.min(aPos)
    
    aNeg =  np.ma.masked_where(c>0, a) # only values where c < 0 
    alphaMin = np.max(aNeg)
    
    # Choose alpha1 or alpha2
    alpha = alphaMin
    
    # Remove Node
    vals = weights <= alpha*c
    print(np.sum(vals))
    idx = np.argmax(vals)
    if (np.sum(vals)) !=1:
        idx = np.argmin(weights - alpha*c)
        print("No w_k is less than  alpha_k*c_k")
    nodes = np.delete(nodes, idx, axis=0)
    
    weights = weights - alpha*c
    weights = np.delete(weights, idx, axis=0)
    print(np.sum(weights))
    
    
plt.figure()
plt.scatter(samples[:,0], samples[:,1], marker='*')
plt.scatter(nodes[:,0], nodes[:,1])
plt.show()


    
  

    
    
    
    
    
    
    
    
    
    
    