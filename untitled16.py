# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 18:28:12 2020

@author: Rylei
"""
import numpy as np
import Functions as fun
import UnorderedMesh as UM
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
import untitled13 as u13

Px = 0
Py = 0
h=0.01

mesh = UM.generateOrderedGridCenteredAtZero(-1, 1, -1, 1, 0.05, includeOrigin=True)
vals = []
for i in range(len(mesh)):
    value = fun.G(Px, Py, mesh[i,0], mesh[i,1], h)
    vals.append(value)
    
muX = Px #+ h*fun.f1(Px,Py)
muY = Py #+ h*fun.f2(Px,Py)
sigmaX = np.sqrt(h)*fun.g1()
sigmaY = np.sqrt(h)*fun.g2()
rv = multivariate_normal([muX, muY], [[sigmaX**2, 0], [0, sigmaX**2]])
gauss = np.asarray([rv.pdf(mesh)]).T

# Lg, tt = u13.calculateLg(Px, Py, mesh, h)
# Lf,ttt = u13.calculateLf(Px, Py, mesh, h)
vals = u13.newIntegrand(Px, Py, mesh, h)
    
fig = plt.figure()
ax = Axes3D(fig)
ax.plot(mesh[:,0], mesh[:,1], np.asarray(vals), '*r', label='new part of integrand')
ax.scatter(mesh[:,0], mesh[:,1], np.squeeze(gauss), c='g', label='Gaussian')
ax.legend()
# ax.scatter(Px, Px, 0, c='r')

