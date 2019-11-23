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

T = 0.01  # final time, code computes PDF of X_T
s = 0.75  # the exponent in the relation k = h^s
h = 0.01  # temporal step size
init = 0  # initial condition X_0
numsteps = int(np.ceil(T / h))

assert numsteps > 0, 'The variable numsteps must be greater than 0'

# define spatial grid
kstep = h ** s
kstep = 0.2
epsilonTol = -5
xmin=-2.2
xmax=2.3
ymin=-2.2
ymax=2.3
h=0.01

def generateGRow(point, allPoints, kstep, h):
    row = []
    for i in range(len(allPoints)):
        val = kstep**2*fun.G(point[0], point[1], allPoints[i,0], allPoints[i,1], h)
        row.append(val)
    return row

def loopNewPDf(Px, Py, grid, kstep, h, interpPDF):
    val = 0
    for i in range(len(interpPDF)):
        val = val + kstep**2*fun.G(Px, Py, grid[i,0], grid[i,1], h)*interpPDF[i]
    return val

mesh = UM.generateOrderedGrid(xmin, xmax, ymin, ymax, kstep)      # ordered mesh  
#mesh = UM.generateRandomPoints(xmin,xmax,ymin,ymax,800)  # unordered mesh
#mesh = np.vstack((mesh,mesh2))


pdf= UM.generateICPDF(mesh[:,0], mesh[:,1], 0.1, 0.1)
#pdf = np.zeros(len(mesh))

#pdf[int(np.sqrt(len(mesh))/2 * np.sqrt(len(mesh)))]=10
#pdf[1000]=10

#fig = plt.figure()
#ax = Axes3D(fig)
#ax.scatter(mesh[:,0], mesh[:,1], pdf, c='r', marker='.')

Meshes = []
PdfTraj = []
PdfTraj.append(np.copy(pdf))
Meshes.append(np.copy(mesh))


def PrepareMeshInfo():
    GMat = []
    Grids = []
    Vertices = []
    VerticesNum = []
    for point in trange(len(mesh)):
        tri = Delaunay(mesh)
        grid = UM.makeOrderedGridAroundPoint([mesh[point,0],mesh[point,1]],kstep, max(xmax-xmin, ymax-ymin), xmin,xmax,ymin,ymax)
        Grids.append(np.copy(grid))
        Vertices.append([])
        VerticesNum.append([])
        for currGridPoint in range(len(grid)):
            vertices, indices = UM.getVerticesForPoint([grid[currGridPoint,0], grid[currGridPoint,1]], mesh, tri) # Points that make up triangle
            Vertices[point].append(np.copy(vertices))
            VerticesNum[point].append(np.copy(indices))
        gRow = generateGRow([mesh[point,0], mesh[point,1]], grid, kstep, h)
        GMat.append(np.copy(gRow))
    return tri, Grids, Vertices, VerticesNum, GMat
   
tri, Grids, Vertices, VerticesNum, GMat = PrepareMeshInfo()
pdf = np.copy(PdfTraj[-1])
for i in trange(30):
    for point in range(len(mesh)):
        interpPdf = []
        #grid = UM.makeOrderedGridAroundPoint([mesh[point,0],mesh[point,1]],kstep, 3, xmin,xmax,ymin,ymax)
        grid = Grids[point]
        for g in range(len(grid)):
            Px = grid[g,0] # (Px, Py) point to interpolate
            Py = grid[g,1]
            vertices = Vertices[point][g]
                    
            PDFVals = UM.getPDFForPoint(pdf, VerticesNum[point][g])
            interp = UM.baryInterp(Px, Py, vertices, PDFVals)
            interpPdf.append(interp)
        #gRow = generateGRow([mesh[point,0], mesh[point,1]], grid, kstep, h)
        gRow = GMat[point]
        newval = np.matmul(np.asarray(gRow), np.asarray(interpPdf))
        #newval2 = loopNewPDf(mesh[point,0], mesh[point,1], grid, kstep, h, interpPdf)
        pdf[point] = np.copy(newval)
    PdfTraj.append(np.copy(pdf))
    Meshes.append(np.copy(mesh))
    if i >= 3:
            possibleZeros = MeshUp.checkIntegrandForZeroPoints(GMat,pdf, 10**(-8))
            GMat, mesh, Grids, Vertices, VerticesNum, pdf, ChangedBool = MeshUp.removePointsFromMesh(possibleZeros,GMat, mesh, Grids, Vertices, VerticesNum, pdf)
            if ChangedBool ==1:
                tri, Grids, Vertices, VerticesNum, GMat = PrepareMeshInfo()
            xMin = min
t=0
#Use to check triangularization
#plt.plot(mesh[:,0], mesh[:,1], '.k')
#plt.plot(vertices[0,0], vertices[0,1], '.g', markersize=14)
#plt.plot(vertices[1,0], vertices[1,1], '.g',  markersize=14)
#plt.plot(vertices[2,0], vertices[2,1], '.g',  markersize=14)
#plt.plot(Px, Py, '.r', markersize=14)
#plt.show()
##
#fig = plt.figure()
#ax = Axes3D(fig)
#ax.scatter(grid[:,0], grid[:,1], interpPdf, c='r', marker='.')

#fig = plt.figure()
#ax = Axes3D(fig)
#ax.scatter(mesh[:,0], mesh[:,1], PdfTraj[-1], c='r', marker='.')
#    
#    
def update_graph(num):
    graph.set_data (Meshes[num][:,0], Meshes[num][:,1])
    graph.set_3d_properties(PdfTraj[num])
    title.set_text('3D Test, time={}'.format(num))
    return title, graph, 


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
title = ax.set_title('3D Test')
    
graph, = ax.plot(Meshes[-1][:,0], Meshes[-1][:,1], PdfTraj[-1], linestyle="", marker="o")
ax.set_zlim(0, np.max(PdfTraj[15]))
ani = animation.FuncAnimation(fig, update_graph, frames=len(PdfTraj),
                                         interval=1000, blit=False)

plt.show()
