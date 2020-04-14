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
import sys
sys.path.insert(1, r'C:\Users\Rylei\Documents\SimpleDTQ\pyopoly1')
from families import HermitePolynomials
import indexing
import LejaPoints as LP
import MeshUpdates2D as meshUp


poly = HermitePolynomials(rho=0)
d=2
k = 40    
ab = poly.recurrence(k+1)
lambdas = indexing.total_degree_indices(d, k)
poly.lambdas = lambdas


mesh, two = LP.getLejaPoints(30, np.asarray([[0,0]]).T, poly, candidateSampleMesh = [], returnIndices = False)
mesh = LP.mapPointsBack(0, 0, mesh, np.sqrt(0.005), np.sqrt(0.005))

mesh2, two = LP.getLejaPoints(130, np.asarray([[0,0]]).T, poly, candidateSampleMesh = [], returnIndices = False)
mesh2 = LP.mapPointsBack(0, 0, mesh2, 0.1, 0.1)

meshFinal = np.copy(mesh)
for i in range(len(mesh2)):
    meshn = LP.mapPointsBack(mesh2[i,0], mesh2[i,1], mesh,1, 1)
    meshFinal = np.vstack((meshFinal,np.copy(meshn)))


plt.figure()
plt.scatter(meshFinal[:,0], meshFinal[:,1])
plt.scatter(mesh2[:,0], mesh2[:,1], c='r')