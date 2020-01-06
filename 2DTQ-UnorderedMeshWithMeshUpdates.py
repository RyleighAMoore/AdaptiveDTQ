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
kstep = 0.15
epsilonTol = -5
xmin=-3.
xmax=3.
ymin=-3.
ymax=3.
h=0.01


#def loopNewPDf(Px, Py, grid, kstep, h, interpPDF):
#    val = 0
#    for i in range(len(interpPDF)):
#        val = val + kstep**2*fun.G(Px, Py, grid[i,0], grid[i,1], h)*interpPDF[i]
#    return val

mesh = UM.generateOrderedGridCenteredAtZero(-1.5, 1.5, -1.5, 1.5, kstep)      # ordered mesh  
#mesh = UM.generateRandomPoints(xmin,xmax,ymin,ymax,200)  # unordered mesh
#circle_r = 0.7
## center of the circle (x, y)
#circle_x = 0
#circle_y = 0
#mesh2 = []
#for r in range(100):
#    # random angle
#    alpha = 2 * math.pi * random.random()
#    # random radius
#    r = circle_r * math.sqrt(random.random())
#    # calculating coordinates
#    x = r * math.cos(alpha) + circle_x
#    y = r * math.sin(alpha) + circle_y
#    mesh2.append([x, y])

#mesh2 = UM.generateOrderedGrid(-0.25, 0.25, -0.25, 0.25, 0.08)      # ordered mesh  
#mesh = np.vstack((mesh,mesh2))

#mesh = np.vstack((mesh,[0,0]))
#mesh = np.vstack((mesh,[0.05,0.05]))
#mesh = np.vstack((mesh,[-0.05,0.05]))
#mesh = np.vstack((mesh,[0.05,-0.05]))
#mesh = np.vstack((mesh,[-0.05,-0.05]))

#pdf = np.zeros(len(mesh))
#mesh = mesh2
pdf = UM.generateICPDF(mesh[:,0], mesh[:,1], 0.5, 0.5)
#pdf = np.zeros(len(mesh))
#pdf[-1]=10
#pdf[1000]=10


Meshes = []
PdfTraj = []
PdfTraj.append(np.copy(pdf))
Meshes.append(np.copy(mesh))
GMat = []
Grids = []
Vertices = []
VerticesNum = []

tri = Delaunay(mesh, incremental=True)
for point in trange(len(mesh)):
    grid = UM.makeOrderedGridAroundPoint([mesh[point,0],mesh[point,1]],kstep, max(xmax-xmin, ymax-ymin), xmin,xmax,ymin,ymax)
    Grids.append(np.copy(grid))
    Vertices.append([])
    VerticesNum.append([])
    for currGridPoint in range(len(grid)):
        vertices, indices = UM.getVerticesForPoint([grid[currGridPoint,0], grid[currGridPoint,1]], mesh, tri) # Points that make up triangle
        Vertices[point].append(np.copy(vertices))
        VerticesNum[point].append(np.copy(indices))
    gRow = MeshUp.generateGRow([mesh[point,0], mesh[point,1]], grid, kstep, h)
    GMat.append(np.copy(gRow))


#def PrepareVerticesInfo(mesh):
#    for point in trange(len(mesh)):
#        Vertices.append([])
#        VerticesNum.append([])
#        for currGridPoint in range(len(grid)):
#            vertices, indices = UM.getVerticesForPoint([grid[currGridPoint,0], grid[currGridPoint,1]], mesh, tri) # Points that make up triangle
#            Vertices[point].append(np.copy(vertices))
#            VerticesNum[point].append(np.copy(indices))
#    return Vertices, VerticesNum      
    
pdf = np.copy(PdfTraj[-1])
adjustGrid = False
for i in trange(5):
    if (i >= 0) and adjustGrid:
            #possibleZerosIntegral = MeshUp.checkIntegralForZeroPoints(GMat, pdf, 10**(-10))
#            Zeros = [possibleZerosIntegrand + possibleZerosIntegral == 2]
###################################################### Check if remove points
            GMat, mesh, Grids, pdf, remBool = MeshUp.removePointsFromMesh(GMat, mesh, Grids, Vertices, VerticesNum, pdf, tri, True)
######################################################
            if (remBool == 1):
                tri = Delaunay(mesh, incremental=True)
                Vertices = []
                VerticesNum = []
                for point in range(len(mesh)): # Recompute Vertices and VerticesNum matrices
                    grid = Grids[point]
                    Vertices.append([])
                    VerticesNum.append([])
                    for currGridPoint in range(len(grid)):
                        vertices, indices = UM.getVerticesForPoint([grid[currGridPoint,0], grid[currGridPoint,1]], mesh, tri) # Points that make up triangle
                        Vertices[point].append(np.copy(vertices))
                        VerticesNum[point].append(np.copy(indices))
#           
            mesh, GMat, Grids, Vertices, VerticesNum, pdf, tri, addBool = MeshUp.addPointsToMesh(mesh, GMat, Grids, Vertices, VerticesNum, pdf, tri, kstep, h, xmin, xmax, ymin, ymax)
            if (addBool == 1):
                Vertices = []
                VerticesNum = []
                for point in range(len(mesh)): # Recompute Vertices and VerticesNum matrices
                    grid = Grids[point]
                    Vertices.append([])
                    VerticesNum.append([])
                    for currGridPoint in range(len(grid)):
                        vertices, indices = UM.getVerticesForPoint([grid[currGridPoint,0], grid[currGridPoint,1]], mesh, tri) # Points that make up triangle
                        Vertices[point].append(np.copy(vertices))
                        VerticesNum[point].append(np.copy(indices))
                        
    print("stepping forward")
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
    print(len(mesh))
    
    
            
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

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(Meshes[0][:,0], Meshes[0][:,1], PdfTraj[0], c='r', marker='.')
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
ax.set_zlim(0, np.max(PdfTraj[2]))
ani = animation.FuncAnimation(fig, update_graph, frames=len(PdfTraj),
                                         interval=1000, blit=False)

plt.show()
