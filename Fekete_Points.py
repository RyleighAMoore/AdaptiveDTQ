# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 12:28:45 2020

@author: Ryleigh
"""

import numpy.polynomial.polynomial as poly
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import UnorderedMesh as UM
import numpy.polynomial.polynomial as poly2d
# Returns and graphs the amount of numNewPoints Fekete points on [a,b]
# using the monomial basis functions.
def oneDFekete(a,b, numNewPoints, numTimeOrthogonalize):
    points = np.linspace(a,b,5001)
    numDesiredPoints = numNewPoints
    V = np.ones([len(points), numDesiredPoints])
    for i in range(len(points)): #rows
        for j in range(numDesiredPoints): #cols
            V[i,j] = points[i]**j
    
    Vorig = V
    T  = np.identity(numDesiredPoints)
    for s in range(numTimeOrthogonalize): #Sucessive orthogonalization Usually s ranges from 0 to 1
        print("Orthogonalization")
        Q, R = np.linalg.qr(V)
        P = np.linalg.inv(R)
        V = np.matmul(V,P)
        T = np.matmul(T,P)
        
    Q, R, perm = sp.linalg.qr(V.T, pivoting=True)
    newPoints = points[perm[:numDesiredPoints]]
    plt.scatter(newPoints,np.zeros(numDesiredPoints), s=50)
    return newPoints

#newpoints = oneDFekete(-1,1,5,1)
    
def twoDFeketeSquare(a,b,stepSize, numPointsDesired):
    points = UM.generateOrderedGridCenteredAtZero(-a, b, -a, b, stepSize)      # ordered mesh  
    numDesiredPoints = numPointsDesired
    
    inds = np.asarray(list(range(0, numDesiredPoints*numDesiredPoints)))
    inds_unrav = np.unravel_index(inds, (numDesiredPoints, numDesiredPoints))
    newIndex = []
    
    for val in range(np.size(inds_unrav[0])):
        t = inds_unrav[1][val]*(numDesiredPoints + 1)+inds_unrav[0][val]
        newIndex.append(t) 
    
    V = np.ones([len(points), numDesiredPoints])
    for i in range(len(points)):
            V[i,1] = points[:,0][i] #x
            V[i,2] = points[:,1][i] #y
            V[i,3] = points[:,0][i]*points[:,1][i] #xy
            V[i,4] = points[:,0][i]**2 #x^2
            V[i,5] = points[:,1][i]**2 #y^2
            V[i,6] = points[:,0][i]**2*points[:,1][i] #x^2y
            V[i,7] = points[:,0][i]*points[:,1][i]**2 #xy^2
            V[i,8] = points[:,0][i]**3 #x^3
            V[i,9] = points[:,1][i]**3 #y^3
    
    for i in range(len(points)):
            V[i,9] = points[:,1][i]**3 #y^3
            
    #V = np.ones([len(points), numDesiredPoints])
    #for i in range(len(points)): #rows
    #    for j in range(numDesiredPoints): #cols
    #        ind = newIndex.index(j)
    #        V[i,j] = points[:,0][i]**inds_unrav[0][ind]*points[:,1][i]**inds_unrav[1][ind]
    
    Vorig = V
    T  = np.identity(numDesiredPoints)
    for s in range(1): #Sucessive orthogonalization Usually s ranges from 0 to 1
        print("Orthogonalization")
        Q, R = np.linalg.qr(V)
        P = np.linalg.inv(R)
        V = np.matmul(V,P)
        T = np.matmul(T,P)
        
    Q, R, perm = sp.linalg.qr(V.T, pivoting=True)
    newPointsX = points[:,0][perm[:numDesiredPoints]]
    newPointsY = points[:,1][perm[:numDesiredPoints]]
    plt.scatter(newPointsX,newPointsY, s=50)

#twoDFeketeSquare(1,1,0.001, 10)
    
    


