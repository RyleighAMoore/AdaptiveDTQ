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
import pickle
import LejaQuadrature as LQ
import distanceMetrics as DM
import sys
sys.path.insert(1, r'C:\Users\Rylei\Documents\SimpleDTQ\pyopoly1')
from families import HermitePolynomials
import indexing
import LejaPoints as LP
import MeshUpdates2D as meshUp
from Scaling import GaussScale

def getICMesh(radius):
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
    
    mesh, two = LP.getLejaPoints(130, np.asarray([[0,0]]).T, poly, candidateSampleMesh = [], returnIndices = False)
    mesh = LP.mapPointsBack(0, 0, mesh, np.sqrt(0.005), np.sqrt(0.005))
    
    mesh2, two = LP.getLejaPoints(130, np.asarray([[0,0]]).T, poly, candidateSampleMesh = [], returnIndices = False)
    mesh2 = LP.mapPointsBack(0, 0, mesh2, np.sqrt(h)*fun.g1(), np.sqrt(h)*fun.g2())


    meshSpacing = 0.1 #DM.separationDistance(mesh)*2
    grid = UM.generateOrderedGridCenteredAtZero(-1.6, 1.6, -1.6, 1.6, meshSpacing , includeOrigin=True)
    noise = np.random.normal(0,np.sqrt(h)*fun.g1(), size = (len(grid),2))
    
    noise = np.random.uniform(-meshSpacing, meshSpacing,size = (len(grid),2))
    
    shake = 0
    noise = -meshSpacing*shake +(meshSpacing*shake - - meshSpacing*shake)/(np.max(noise)-np.min(noise))*(noise-np.min(noise))
    noiseGrid = grid+noise
    x,y = noiseGrid.T
    X = []
    Y = []
    # X.append(0)
    # Y.append(0)
    for point in range(len(grid)):
        if np.sqrt(x[point]**2 + y[point]**2) < radius:
            X.append(x[point])
            Y.append(y[point])
    
    newGrid = np.vstack((X,Y))
    x,y = newGrid
    x1,y1 = mesh.T
    x2,y2 = mesh2.T
    
    # plt.figure()
    # plt.scatter(x,y)
    # plt.scatter(x1,y1,c='red')
    # plt.scatter(x2,y2,c='green')
    # plt.show()

    meshSpacing2 = DM.separationDistance(newGrid.T)
    print(len(newGrid.T))
    return newGrid.T


if __name__ == "__main__":
    getICMesh(0.5)