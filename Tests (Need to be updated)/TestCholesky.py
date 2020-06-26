import sys
sys.path.insert(1, r'..')
import ICMeshGenerator as M
import matplotlib.pyplot as plt
from QuadratureUtils import GetGaussianPart
from GaussFit import fitGaussian
from QuadraticFit import fitQuad
from scipy.interpolate import griddata
import Functions as fun
import variableTransformations as VT
import numpy as np
import QuadratureRules as QR
from families import HermitePolynomials
import indexing
import LejaPoints as LP
import opolynd
from mpl_toolkits.mplot3d import Axes3D

poly = HermitePolynomials(rho=0)
d=2
k = 40    
ab = poly.recurrence(k+1)
lambdas = indexing.total_degree_indices(d, k)
poly.lambdas = lambdas

def polyFun(x,y):
    return np.ones(len(x))

#Generate a mesh
mesh1 = M.getICMesh()
pdf1 = polyFun(mesh1[:,0], mesh1[:,1])

#Fit with QuadFit procedure
scale1, temp, cc, Const = fitQuad(0, 0, mesh1, pdf1)
mesh, pdf = LP.getLejaSetFromPoints(scale1, mesh1, 12, poly, pdf1, 0)

#Find L and transform
L = np.linalg.cholesky((scale1.cov))
Linv = np.linalg.inv(L)   

#Get canonical mesh 
stdMesh, two = LP.getLejaPoints(12, np.asarray([[0,0]]).T, poly, candidateSampleMesh = [], returnIndices = False)
mesh = stdMesh

LuMu = (L@mesh.T).T + scale1.mu.T*np.ones(np.shape(mesh.T)).T
u = (Linv@(mesh- scale1.mu.T*np.ones(np.shape(mesh.T)).T).T).T

x,y = LuMu.T
vals = np.exp(-(cc[0]*x**2+ cc[1]*y**2 + 2*cc[2]*x*y + cc[3]*x + cc[4]*y + cc[5]))/Const
    
pdfNew = polyFun(LuMu[:,0], LuMu[:,1]) / vals


#Compute quadrature rule

V = opolynd.opolynd_eval(u, poly.lambdas[:len(u),:], poly.ab, poly)
vinv = np.linalg.inv(V)
c = np.matmul(vinv, pdf)
L = np.linalg.cholesky((scale1.cov))
JacFactor = np.prod(np.diag(L))

plot = False
if plot:
    if np.sum(np.abs(vinv[0,:])) > 0:
        interp = np.matmul(V,c)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(mesh[:,0], mesh[:,1], pdf, c='r', marker='o')
        ax.scatter(u[:,0], u[:,1], pdf, c='k', marker='.')
        
print(c[0]*JacFactor*np.pi)
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(mesh[:,0], mesh[:,1],pdf, c='r', marker='.')
# ax.scatter(u[:,0], u[:,1],pdf, c='k', marker='.')

ans = c[0]*JacFactor*np.pi
cond = np.sum(np.abs(vinv[0,:]))

value, condNum = QR.QuadratureByInterpolationND(poly, scale1, u, pdfNew.T)


plt.figure()
plt.scatter(mesh1[:,0], mesh1[:,1], c='k')
plt.scatter(u[:,0], u[:,1], c='r')




