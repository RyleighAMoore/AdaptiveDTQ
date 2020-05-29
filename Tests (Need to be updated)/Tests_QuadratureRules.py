# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 12:42:08 2020

@author: Rylei
"""
from QuadratureRules import *
from families import *
from indexing import *
import numpy as np
from math import isclose
from LejaPoints import *
import sys
sys.path.insert(1, r'C:\Users\Rylei\Documents\SimpleDTQ')
from Functions import *

from Scaling import GaussScale
'''
Check int -inf to inf of 1/(sqrt(2pi)*0.5)exp(-x^2/(2*0.5^2)) (x^2+x+2) dx = 2.25
using interplatory quadrature and Gauss-Hermite quadrature
'''
def test_Hermite1D():
    H = HermitePolynomials(rho=0)
    mu=0
    sigma=.5
    scale = GaussScale(1)
    scale.setMu([[mu]])
    scale.setCov([[sigma**2]])
    # scaling = np.asarray([[mu, sigma]])
    N=3
    
    mesh, w = H.gauss_quadrature(N)
    # mesh = np.linspace(-1,1, N)
    pdf = mesh**2+mesh+2
    ans1 = QuadratureByInterpolation1D(H, scale, mesh, pdf)
    
    # Compute again using nodes and weights
    mesh, w = H.gauss_quadrature(N)
    mesh = VT.map_from_canonical_space(mesh, scale)
    pdf = mesh**2+mesh+2
    
    ans2 = np.dot(w.T, pdf)
    
    assert isclose(ans1, 2.25)
    assert isclose(ans2, 2.25)
    print("test_Hermite1D - PASS")
    

'''2D example of QuadratureByInterpolation with corrected integrand
#int -inf to inf int -inf to inf of (1/(sqrt(2pi)*0.5) e^(-(x)^2/(2*0.5^2)) 1/(sqrt(2pi)*0.5) e^(-(y)^2/(2*0.5^2))) (x^2+y +3x)dxdy = 0.25
'''
def test_Hermite2D():
    H = HermitePolynomials(rho=0)
    d=2
    k = 6  
    N=6  
    ab = H.recurrence(k+1)
    lambdas = total_degree_indices(d, k)
    H.lambdas = lambdas
    sigmaX = 0.5
    sigmaY = 0.5
    
    scale = GaussScale(2)
    scale.setMu(np.asarray([[0,0]]).T)
    scale.setSigma(np.asarray([sigmaX,sigmaY]))
    
    mesh, two = getLejaPoints(N, np.asarray([[0,0]]).T, H, candidateSampleMesh = [], returnIndices = False)
    
    mesh = mapPointsBack(0, 0, mesh, sigmaX, sigmaY)
    
    pdf = mesh[:,0]**2 + mesh[:,1] + 3*mesh[:,0]
        
    value, condNum = QuadratureByInterpolationND(H, scale, mesh, pdf)
    assert isclose(value, 0.25)
    print("test_Hermite2D - PASS")


    
'''2D example of QuadratureByInterpolation with corrected integrand
#int -inf to inf int -inf to inf of (1/(sqrt(2pi)*0.5) e^(-(x)^2/(2*0.5^2)) 1/(sqrt(2pi)*0.3) e^(-(y)^2/(2*0.3^2))) (x^2+2y^2)dxdy = 0.43
'''
def test_Hermite2D_diffSigma():
    H = HermitePolynomials(rho=0)
    d=2
    k = 6  
    N=6  
    ab = H.recurrence(k+1)
    lambdas = total_degree_indices(d, k)
    H.lambdas = lambdas
    sigmaX = 0.5
    sigmaY = 0.3
    
    scale = GaussScale(2)
    scale.setMu(np.asarray([[0,0]]).T)
    scale.setSigma(np.asarray([sigmaX,sigmaY]))
    
    mesh, two = getLejaPoints(N, np.asarray([[0,0]]).T, H, candidateSampleMesh = [], returnIndices = False)
    
    mesh = mapPointsBack(0, 0, mesh, sigmaX, sigmaY)
    
    pdf = mesh[:,0]**2 + 2*mesh[:,1]**2
    
    value, condNum = QuadratureByInterpolationND(H, scale, mesh, pdf)
    assert isclose(value, 0.43)
    print("test_Hermite2D_diffSigma - PASS")


'''
int -inf to inf int -inf to inf of (1/(sqrt(2pi)*0.5) e^(-(x-0.5)^2/(2*0.5^2)) 1/(sqrt(2pi)*0.3) e^(-(y+0.2)^2/(2*0.3^2))) (x^2+2y^2)dxdy = 0.76
'''
def test_Hermite2D_diffSigma_diffMu():
    H = HermitePolynomials(rho=0)
    d=2
    k = 6  
    N=6  
    ab = H.recurrence(k+1)
    lambdas = total_degree_indices(d, k)
    H.lambdas = lambdas
    sigmaX = 0.5
    sigmaY = 0.3
    
    scale = GaussScale(2)
    scale.setMu(np.asarray([[0.5,-0.2]]).T)
    scale.setSigma(np.asarray([sigmaX,sigmaY]))
    
    mesh, two = getLejaPoints(N, np.asarray([[0,0]]).T, H, candidateSampleMesh = [], returnIndices = False)
    
    mesh = mapPointsBack(0, 0, mesh, sigmaX, sigmaY)
    
    pdf = mesh[:,0]**2 + 2*mesh[:,1]**2
    
    value, condNum = QuadratureByInterpolationND(H, scale, mesh, pdf)
    assert isclose(value, 0.76)
    print("test_Hermite2D_diffSigma_diffMu - PASS")


# test_Hermite1D()
# test_Hermite2D()
# test_Hermite2D_diffSigma()
# test_Hermite2D_diffSigma_diffMu()


from Plotting import *
#Works well for simple H functions
def test_Hermite2D_Gauss_viaHLinCombGauss(N, meshL):
    H = HermitePolynomials(rho=0)
    d=2
    k=50
    N=N  
    ab = H.recurrence(k+1)
    lambdas = total_degree_indices(d, k)
    H.lambdas = lambdas
    sigmaX = 0.1*g1()
    sigmaY = 0.1*g2()
    mesh = meshL[:N,:]
    assert len(mesh) >= N
    
    scale = GaussScale(2)
    scale.setMu(np.asarray([[0,0]]).T)
    scale.setSigma(np.asarray([sigmaX,sigmaY]))
    
    scale0 = GaussScale(2)
    scale0.setMu(np.asarray([[0,0]]).T)
    scale0.setSigma(np.asarray([0.1,0.1]))
    
    pdfO = GVals(0, 0, mesh, 0.01)*Gaussian(0, 0, 0.1, 0.1, mesh)    
    
    pdf = HVals(0, 0, mesh, 0.01)
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(mesh[:,0], mesh[:,1], pdf,  c='r', marker='o')
    
    # scaling, pdf = GetGaussianPart(0, 0, mesh, pdf, 0.01, round=1)

    scaleNew, cfinal = productGaussians2D(scale, scale0)

    value, condNum = QuadratureByInterpolationND(H, scaleNew, mesh, pdf)
    print(value*cfinal)
    
    return value*cfinal


#Tries to divide gaussian out of whole integrand
# def test_Hermite2D_Gauss_viaG(N, meshL):
#     H = HermitePolynomials(rho=0)
#     d=2
#     k=50
#     N=N  
#     ab = H.recurrence(k+1)
#     lambdas = total_degree_indices(d, k)
#     H.lambdas = lambdas
#     sigmaX = 0.1*g1()
#     sigmaY = 0.1*g2()
#     mesh = meshL[:N,:]
#     assert len(mesh) >= N

    
#     pdfO = GVals(0, 0, mesh, 0.01)*Gaussian(0, 0, 0.1, 0.1, mesh)

#     pdf = HVals(0, 0, mesh, 0.01)*Gaussian(0, 0, 0.1, 0.1, mesh)
    
#     scaling, pdf = GetGaussianPart(0, 0, mesh, pdfO, 0.01, round=2)
    
#     pdfNew = pdf*Gaussian(scaling[0,0], scaling[1,0], scaling[0,1], scaling[1,1], mesh)
    
#     scaling = np.asarray([[scaling[0,0], scaling[0,1]], [scaling[1,0],scaling[1,1]]])
        
#     # fig = plt.figure()
#     # ax = Axes3D(fig)
#     # ax.scatter(mesh[:,0], mesh[:,1], pdfO,  c='r', marker='o')
#     # ax.scatter(mesh[:,0], mesh[:,1], pdfNew,  c='k', marker='*')
    
#     # fig = plt.figure()
#     # ax = Axes3D(fig)
#     # ax.scatter(mesh[:,0], mesh[:,1], pdf,  c='r', marker='o')

#     print(scaling)

#     value, condNum = QuadratureByInterpolationND(H, scaling, mesh, pdf)
#     print(value)
#     return value
    # assert isclose(value, 0.76)
    # print("test_Hermite2D_diffSigma_diffMu - PASS")
    
# def test_Hermite2D_Gauss_viaHLinCombGaussTest(N, meshL):
#     H = HermitePolynomials(rho=0)
#     d=2
#     k=50
#     N=N  
#     ab = H.recurrence(k+1)
#     lambdas = total_degree_indices(d, k)
#     H.lambdas = lambdas
#     sigmaX = 0.1*g1()
#     sigmaY = 0.1*g2()
#     mesh = meshL[:N,:]
#     assert len(mesh) >= N
    
#     pdfO = GVals(0, 0, mesh, 0.01)*Gaussian(0, 0, 0.1, 0.1, mesh)    
#     pdfpart = HVals(0, 0, mesh, 0.01)*Gaussian(0, 0, 0.1, 0.1, mesh)   
#     # pdfpart = HVals(0, 0, mesh, 0.01)

#     scaling, pdf = GetGaussianPart(0, 0, mesh, pdfpart, 0.01, round=2)
    
#     print(scaling)
#     muNew, sigmaNew, cfinal1 = productGaussians2D(0, 0, 0, 0, sigmaX, sigmaY, scaling[0,1], scaling[1,1])
#     # muNew, sigmaNew, cfinal2 = productGaussians2D(0, 0, muNew[0][0], muNew[1][0], 0.1, 0.1, sigmaNew[0,0], sigmaNew[1,1])
#     cfinal2=1
    
#     # pdfTest = cfinal1*cfinal2*pdf*Gaussian(0, 0, sigmaNew[0,0], sigmaNew[1,1], mesh)
    
#     # fig = plt.figure()
#     # ax = Axes3D(fig)
#     # ax.scatter(mesh[:,0], mesh[:,1], (pdf),  c='r', marker='o')
#     # ax.scatter(mesh[:,0], mesh[:,1], pdfTest,  c='k', marker='.')
    
    
#     scaling = np.asarray([[muNew[0][0], sigmaNew[0,0]], [muNew[1][0], sigmaNew[1,1]]])
        
#     value, condNum = QuadratureByInterpolationND(H, scaling, mesh, pdf)
#     print(value*cfinal1*cfinal2)
#     return value*cfinal1*cfinal2



H = HermitePolynomials(rho=0)
d=2
k=30
N=300  
ab = H.recurrence(k+1)
lambdas = total_degree_indices(d, k)
H.lambdas = lambdas
sigmaX = 0.08
sigmaY = 0.08

soln= 6.43744
# soln = 7.95775
# soln = 7.39712
# soln = 7.93788
# soln = 4.336
# soln = 7.21461
# soln = 0


# 
Ns = [10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300]
# Ns = [1,10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100,200]

Ns = [5,10,11,12,13,14, 15, 20,25,30,40,100,200,300]
# Ns = [10,30]

Meshes = []
for num in range(1):
    print(num)
    meshL, two = getLejaPoints(300, np.asarray([[0,0]]).T, H, candidateSampleMesh = [], returnIndices = False)
    meshL = mapPointsBack(0, 0, meshL, sigmaX, sigmaY)
    Meshes.append(meshL)

valsMean = []
valsMax = []
valsMin = []
vvs = []
for i in Ns:
    vv = []
    for j in range(len(Meshes)):
        # print(i)
        v = test_Hermite2D_Gauss_viaHLinCombGauss(i, Meshes[j])
        vv.append(v)
        # if np.abs(v-7.95775) < 0.01:
        #     plt.figure()
        #     plt.scatter(Meshes[j][:i,0], Meshes[j][:i,1])
        #     plt.show()
    vv = np.asarray(vv)
    vvs.append(vv)
    valsMean.append(np.mean(np.abs(vv-soln)))
    valsMax.append(np.max(np.abs(vv-soln)))
    valsMin.append(np.min(np.abs(vv-soln)))

plt.figure()
plt.loglog(Ns, np.abs(np.asarray(valsMean)))
plt.loglog(Ns, np.abs(np.asarray(valsMin)))
plt.loglog(Ns, np.abs(np.asarray(valsMax)))
plt.show()

plt.figure()
for t in range(len(vvs)):
    plt.loglog(Ns[t]*np.ones(len(vv)), np.abs(vvs[t]-soln),'.')
plt.loglog(Ns, np.abs(np.asarray(valsMean)),label = "Mean Error")
plt.loglog(Ns, np.abs(np.asarray(valsMin)),label = "Min Error")
plt.loglog(Ns, np.abs(np.asarray(valsMax)),label = "Max Error")
plt.legend()
# plt.ylim([10**(-10), 0])
plt.show()



 