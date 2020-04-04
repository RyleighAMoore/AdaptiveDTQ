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
    
    # import getPCE as PCE
    # import GenerateLejaPoints as LP
    import UnorderedMesh as UM
    from LejaPoints import getLejaPoints,mapPointsBack


    def fun1D(mesh):
        return 2*mesh**2+4
    
    def fun2D(mesh):
        
        # return np.ones(len(mesh))
        return 2*mesh[:,0]*mesh[:,1]**2 + 4
   
    '''
    1D example of QuadratureByInterpolation
    '''
    # H = HermitePolynomials(rho=0)
    # mu=0
    # sigma=.1
    # scaling = np.asarray([[mu, sigma]])
    # N=4
    # # mesh, w = H.gauss_quadrature(N)
    # mesh = np.linspace(-1,1, N)
    # pdf = fun1D(mesh)
    # QuadratureByInterpolation1D(H, scaling, mesh, pdf)
    ''''''
    
    '''
    2D example of QuadratureByInterpolation
    '''
    # d=2
    # k = 40    
    # ab = H.recurrence(k+1)
    # lambdas = indexing.total_degree_indices(d, k)
    
    # N = 5
    # # x = np.linspace(-5,5, N)
    # mesh, two = getLejaPoints(36, np.asarray([[0,0]]).T, H, lambdas, ab, candidateSampleMesh = [], returnIndices = False)
    # scaling = np.asarray([[0, 1], [0, 1]])
    # pdf = fun2D(mesh)
    
    # QuadratureByInterpolationND(H, scaling, mesh, pdf, lambdas, ab)
    ''''''
    
    
    def fun2D(mesh):
        # return np.ones(len(mesh))
        # return np.exp(5*mesh[:,0]**2)
        return 1/UM.generateICPDF(mesh[:,0], mesh[:,1], .1, .1)

    '''
    2D example of QuadratureByInterpolation
    '''
    H = HermitePolynomials(rho=0)
    d=2
    k = 40    
    ab = H.recurrence(k+1)
    lambdas = indexing.total_degree_indices(d, k)
    
    # x = np.linspace(-5,5, N)
    mesh, two = getLejaPoints(40, np.asarray([[0,0]]).T, H, lambdas, ab, candidateSampleMesh = [], returnIndices = False)
    mesh = mapPointsBack(0, 0, mesh, .01, .01)
    scaling = np.asarray([[0, .01], [0, .01]])
    pdf = fun2D(mesh)
    QuadratureByInterpolationND(H, scaling, mesh, pdf, lambdas, ab)
    ''''''
    
    # inp = np.hstack((mesh,np.expand_dims(pdf,1)))
    # first = np.asarray([[0],[0]])
    # pdf3 = np.expand_dims(pdf,0)
    # # inp2 = np.vstack(([[0],[0]],pdf3))
    # meanX = np.mean(mesh[:,0])
    # meanY = np.mean(mesh[:,1])
    # meanZ = np.mean(pdf)
    
    # a = inp[:,0] - meanX
    # b = inp[:,1] - meanY
    # c = pdf - meanZ
    # M = np.vstack((a,b))
    # S1 = (1/(len(a)-1))*np.matmul(M,M.T)
    
    # M = np.vstack((b,c))
    # S2 = (1/(len(a)-1))*np.matmul(M,M.T)
    
    # covar = np.cov(inp)
    
    
    
    