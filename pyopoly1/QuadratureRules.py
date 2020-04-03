# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 12:44:33 2020

@author: Rylei
"""
import variableTransformations as VT
import numpy as np
import matplotlib.pyplot as plt
import opolynd
from mpl_toolkits.mplot3d import Axes3D



def QuadratureByInterpolation1D(poly, scaling, mesh, pdf):
    xCan=VT.map_to_canonical_space(mesh, scaling) 
    V = poly.eval(xCan, range(len(xCan)))
    vinv = np.linalg.inv(V)
    c = np.matmul(vinv, pdf)
    print(c[0])
    plot=False
    if plot:
        interp = np.matmul(V,c)
        plt.figure()
        plt.plot(mesh, interp,'.')
        plt.plot(mesh, pdf)
        
def QuadratureByInterpolationND(poly, scaling, mesh, pdf, lambdas, ab):
    numVars = np.size(mesh,0)
    xCan=VT.map_to_canonical_space(mesh, scaling)
    numSamples = len(xCan)                 
    V = opolynd.opolynd_eval(xCan, lambdas[:numSamples,:], ab, poly)
    vinv = np.linalg.inv(V)
    c = np.matmul(vinv, pdf)
    print(c[0])
    plot = False
    if plot:
        interp = np.matmul(V,c)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(mesh[:,0], mesh[:,1], pdf, c='r', marker='o')
        ax.scatter(mesh[:,0], mesh[:,1], interp, c='k', marker='.')
        

if __name__ == "__main__":
    from families import HermitePolynomials
    import indexing
    import sys
    sys.path.insert(1, r'C:\Users\Rylei\Documents\SimpleDTQ')
    
    import getPCE as PCE
    import GenerateLejaPoints as LP
    import UnorderedMesh as UM


    def fun1D(mesh):
        return 2*mesh**2+4
    
    def fun2D(mesh):
        # return np.ones(len(mesh))
        return 2*mesh[:,0]*mesh[:,1]**2 + 4
   
    '''
    1D example of QuadratureByInterpolation
    '''
    H = HermitePolynomials(rho=0)
    mu=0
    sigma=.1
    scaling = np.asarray([[mu, sigma]])
    N=4
    # mesh, w = H.gauss_quadrature(N)
    mesh = np.linspace(-1,1, N)
    pdf = fun1D(mesh)
    QuadratureByInterpolation1D(H, scaling, mesh, pdf)
    
    
    '''
    2D example of QuadratureByInterpolation
    '''
    d=2
    k = 40    
    ab = H.recurrence(k+1)
    lambdas = indexing.total_degree_indices(d, k)
    
    N = 5
    # x = np.linspace(-5,5, N)
    x, w = H.gauss_quadrature(N)
    X,Y = np.meshgrid(x,x)
    mesh = np.concatenate((X.reshape(X.size,1), Y.reshape(Y.size,1)), axis=1)
    poly = PCE.generatePCE(30)
    mesh, mesh2 = LP.getLejaPointsWithStartingPoints([0,0,.1,.1], 10, 5000, poly)

    scaling = np.asarray([[0, 1], [0, 1]])
    pdf = fun2D(mesh)
    
    QuadratureByInterpolationND(H, scaling, mesh, pdf, lambdas, ab)
    
    
    
    
    
    
    
    