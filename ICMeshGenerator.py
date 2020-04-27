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
from Scaling import GaussScale


# define spatial grid
kstep = 0.1
xmin=-2
xmax=2
ymin=-2
ymax=2
h=0.01

IC= np.sqrt(h)*fun.g2()

poly = HermitePolynomials(rho=0)
d=2
k = 40    
ab = poly.recurrence(k+1)
lambdas = indexing.total_degree_indices(d, k)
poly.lambdas = lambdas

mesh, two = LP.getLejaPoints(10, np.asarray([[0,0]]).T, poly, candidateSampleMesh = [], returnIndices = False)
mesh = LP.mapPointsBack(0, 0, mesh, np.sqrt(h)*fun.g1(), np.sqrt(h)*fun.g2())
mesh2 = LP.mapPointsBack(0, 0, mesh, np.sqrt(h)*fun.g1()*0.75, np.sqrt(h)*fun.g2()*0.75)



meshSpacing = DM.separationDistance(mesh2)
for point in range(len(mesh)):
    print(point)
    mesh2 = LP.mapPointsBack(mesh[point,0], mesh[point,1], mesh, np.sqrt(h)*fun.g1()/2, np.sqrt(h)*fun.g2()/2)
    pointsToAdd = meshUp.checkIfDistToClosestPointIsOk(mesh2, mesh, meshSpacing)
    mesh = np.append(mesh, np.asarray(pointsToAdd), axis=0)



