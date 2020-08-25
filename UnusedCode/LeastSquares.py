

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
from families import HermitePolynomials
import indexing
import LejaPoints as LP
import MeshUpdates2D as meshUp
from Scaling import GaussScale
import opoly1d
import opolynd


# H = HermitePolynomials(rho=0)
# d=1
# k = 40    
# ab = H.recurrence(k+1)
# lambdas = indexing.total_degree_indices(d, k)
# H.lambdas = lambdas

# c = H.canonical_connection(4)
# x = np.linspace(-1,1,100)

# basisMat = H.eval(x, range(4))
# C= np.linalg.inv(c)
# vals = np.matmul(basisMat,C.T)

# plt.plot(x,vals[:,:10])



H = HermitePolynomials(rho=0)
d=2
k = 3  
ab = H.recurrence(k+1)
lambdas = indexing.total_degree_indices(d, k)
H.lambdas = lambdas

c = H.canonical_connection(len(lambdas))
x, two = LP.getLejaPoints(6, np.asarray([[0,0]]).T, H, candidateSampleMesh = [], returnIndices = False)
# x = UM.generateOrderedGridCenteredAtZero(-1, 1, -1, 1, 0.1, includeOrigin=True)
pdf = UM.generateICPDF(x[:,0], x[:,1], 1,1)


A = opolynd.opolynd_eval(x, lambdas, ab, H).T
ATA = np.matmul(A.T,A)
# soln = np.matmul(ATA,np.linalg.inv(A.T))


soln = np.linalg.lstsq(ATA, np.log(pdf))[0]
# 


fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x[:,0],x[:,1],soln)
ax.scatter(x[:,0],x[:,1],np.log(pdf), c='red')









