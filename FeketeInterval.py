# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 12:28:45 2020

@author: Ryleigh
"""

import numpy.polynomial.polynomial as poly
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp


points = np.linspace(-1,1,5001)
numDesiredPoints = 50
V = np.ones([len(points), numDesiredPoints])
for i in range(len(points)): #rows
    for j in range(numDesiredPoints): #cols
        V[i,j] = points[i]**j

Vorig = V
T  = np.identity(numDesiredPoints)
for s in range(0): #Sucessive orthogonalization Usually s ranges from 0 to 1
    print("Orthogonalization")
    Q, R = np.linalg.qr(V)
    P = np.linalg.inv(R)
    V = np.matmul(V,P)
    T = np.matmul(T,P)
    
Q, R, perm = sp.linalg.qr(V.T, pivoting=True)
newPoints = points[perm[:numDesiredPoints]]


plt.plot(newPoints,np.zeros(numDesiredPoints), '.')

    

