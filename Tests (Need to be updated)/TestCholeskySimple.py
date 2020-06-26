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
from Scaling import GaussScale


poly = HermitePolynomials(rho=0)
d=2
k = 40    
ab = poly.recurrence(k+1)
lambdas = indexing.total_degree_indices(d, k)
poly.lambdas = lambdas

def polyFun(x,y):
    return y**2
    return np.ones(len(x))


scale1 = GaussScale(2)
scale1.setCov(np.asarray([[1,0.5],[0.5, 1]]))
scale1.setMu(np.asarray([[0.2],[0.45]]))

# Lam, U = np.linalg.eigh(np.linalg.inv(scale1.cov))
# assert all(Lam) > 0
# B = scale1.mu

# La = np.diag(Lam)
# mu = -1/2*U @ np.linalg.inv(La) @ (B.T @ U).T
# Const = np.exp(-0+1/4*B.T@U@np.linalg.inv(La)@U.T@B)




#Find L and transform
L = np.linalg.cholesky((scale1.cov))
Linv = np.linalg.inv(L)   

#Get canonical mesh 
stdMesh, two = LP.getLejaPoints(130, np.asarray([[0,0]]).T, poly, candidateSampleMesh = [], returnIndices = False)
stdMesh = LP.mapPointsBack(scale1.mu[0][0], scale1.mu[1][0], stdMesh, 1, 1)
# stdMesh = LP.mapPointsBack(10, 10, stdMesh, 1,1)


mesh = stdMesh
pdf = polyFun(mesh[:,0], mesh[:,1])

# LuMu = (L@mesh.T).T + scale1.mu.T*np.ones(np.shape(mesh.T)).T
u = (Linv@(mesh - scale1.mu.T*np.ones(np.shape(mesh.T)).T).T).T

# pdfNew = polyFun(LuMu[:,0], LuMu[:,1])


#Compute quadrature rule
V = opolynd.opolynd_eval(u, poly.lambdas[:len(u),:], poly.ab, poly)
vinv = np.linalg.inv(V)
c = np.matmul(vinv, pdf)


L = np.linalg.cholesky((scale1.cov))
JacFactor = np.prod(np.diag(L))

# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(mesh[:,0], mesh[:,1],pdf, c='r', marker='.')
# ax.scatter(u[:,0], u[:,1],pdf, c='k', marker='.')

ans = c[0]*JacFactor*np.pi
cond = np.sum(np.abs(vinv[0,:]))
print(ans)

# scale = GaussScale(2)
# scale.setMu(scale1.mu)
# scale.setCov(np.asarray([[1,0],[0, 1]]))

# value, condNum = QR.QuadratureByInterpolationND(poly, scale, LuMu, pdf.T)
# value, condNum = QR.QuadratureByInterpolationND(poly, scale, u, pdf.T)


# value1, condNum1 = QR.QuadratureByInterpolationND(poly, scale, mesh, pdf.T)


# print(ans,cond)
# print(value, condNum)
# print(value1,condNum1)

# plt.figure()
# plt.scatter(mesh1[:,0], mesh1[:,1], c='k')
# plt.scatter(u[:,0], u[:,1], c='r')




