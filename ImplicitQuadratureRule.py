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
from sympy import Matrix



polyLarge = PCE.generatePCE(50, muX=0, muY=0, sigmaX = 1, sigmaY=1)
poly = PCE.generatePCE(15, muX=0, muY=0, sigmaX = 1, sigmaY=1)
D = len(poly.indices.T)
Kmax = len(polyLarge.indices.T)-1-40 # Will give Kmax+1 samples

samples, mesh2 = LP.getLejaPointsWithStartingPoints([0,0,1,1], Kmax+1, 5000, polyLarge)
plt.scatter(samples[:,0], samples[:,1])

initSamples = samples[:D+1,:]
otherSamples = np.ndarray.tolist(samples[D+1:,:])

vmat = poly.basis_matrix(samples.T).T

weights = np.asarray([(1/(D+1))*np.ones(len(initSamples))]).T
# rv = multivariate_normal([0, 0], [[1, 0], [0, 1]])
# weights = np.asarray([rv.pdf(initSamples)]).T

print(np.sum(weights))
nodes = np.copy(initSamples)
for K in range(D, Kmax): # up to Kmax - 1
    print(K)
    # Add Node
    nodes = np.vstack((nodes, otherSamples.pop()))
    one = ((K+1)/(K+2))*weights
    two = np.asarray([[1/(K+2)]])
    weights = np.concatenate((one, two))
    # weights = np.asarray([rv.pdf(nodes)]).T 
    
    # Update weights
    vmat = poly.basis_matrix(nodes.T).T
    nullspace = sp.linalg.null_space(vmat)



    c = np.asarray([nullspace[:,0]]).T
    
    a = weights/c
    aPos = np.ma.masked_where(c<0, a) # only values where c > 0 
    alpha1 = np.min(aPos.compressed())
    
    aNeg =  np.ma.masked_where(c>0, a) # only values where c < 0 
    alpha2 = np.max(aNeg.compressed())
    
    # Choose alpha1 or alpha2
    alpha = alpha2
    
    # Remove Node
    vals = weights <= alpha*c
    print(np.min(weights - alpha1*c))
    assert np.isclose(np.min(weights - alpha1*c),0)
    print(np.sum(vals))
    idx = np.argmax(vals)
    if (np.sum(vals)) !=1:
        idx = np.argmin(weights - alpha*c)
        print("No w_k is less than  alpha_k*c_k", np.min(weights - alpha*c))
    print(alpha1, alpha2)
    assert alpha2 < alpha1
    nodes = np.delete(nodes, idx, axis=0)
    
    weights = weights - alpha*c
    assert weights[idx] < 10**(-15)
    weights = np.delete(weights, idx, axis=0)
    print(np.sum(weights))
    
    
plt.figure()
plt.scatter(samples[:,0], samples[:,1], marker='*', label = 'Samples')
plt.scatter(nodes[:,0], nodes[:,1], label='Chosen Mesh')
plt.legend()
plt.show()

sigma = 0.1
var = sigma**2

rv = multivariate_normal([0,0], [[var, 0], [0, var]])
vals1 = np.asarray([rv.pdf(nodes)])

print(np.dot(vals1, weights))
np.dot(nodes[:,0]**4, weights)
np.dot(3*nodes[:,0]*np.ones(len(nodes)), weights)


fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(nodes[:,0], nodes[:,1], vals1, c='r', marker='.')

  


plt.scatter(np.reshape(nodes[:,0],-1), np.reshape(nodes[:,1],-1), c=np.reshape(weights,-1), s=300, cmap="summer", edgecolor="k")
plt.colorbar(label="values")

plt.show()
    
    

    
    
    
    
    