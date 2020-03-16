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
import InterpolationPCE as IPCE


T = 0.01  # final time, code computes PDF of X_T
s = 0.75  # the exponent in the relation k = h^s
h = 0.1  # temporal step size
init = 0  # initial condition X_0
numsteps = int(np.ceil(T / h))

assert numsteps > 0, 'The variable numsteps must be greater than 0'

# define spatial grid
kstep = h ** s
kstep = 0.1
xmin=-2
xmax=2
ymin=-2
ymax=2
h=0.01


# pkl_file = open("C:/Users/Rylei/Documents/SimpleDTQ-LejaMesh.p", "wb" )  
# pickle.dump(mesh, pkl_file)
# pkl_file.close()

# pickle_in = open("C:/Users/Rylei/Documents/SimpleDTQ/PickledData/MeshesBimodal.p","rb")
# mesh = pickle.load(pickle_in)
# mesh = mesh[-1]

#def loopNewPDf(Px, Py, grid, kstep, h, interpPDF):
#    val = 0
#    for i in range(len(interpPDF)):
#        val = val + kstep**2*fun.G(Px, Py, grid[i,0], grid[i,1], h)*interpPDF[i]
#    return val

# mesh = UM.generateOrderedGridCenteredAtZero(xmin, xmax, xmin, xmax, kstep, includeOrigin=True)      # ordered mesh  
# mesh = LP.generateLejaMesh(350, .1, .1, 30)
# mesh2 = LP.generateLejaMesh(400, .2, .2, 30)
# mesh2 = UM.generateRandomPoints(-1,1,-1,1,200)  # unordered mesh
# plt.figure()
# plt.plot(mesh[:,0], mesh[:,1], '.r')
# plt.plot(mesh2[:,0], mesh2[:,1], '.r')

# mesh = np.vstack((mesh,mesh2))
# mesh = UM.generateOrderedGridCenteredAtZero(xmin, xmax, xmin, xmax, kstep, includeOrigin=True)


# pdf = UM.generateICPDF(mesh[:,0], mesh[:,1], .1, .1)

# fig = plt.figure()
# ax = Axes3D(fig)
# index =0
# ax.scatter(mesh[:,0], mesh[:,1], pdf, c='r', marker='.')


# plt.figure()
# plt.plot(mesh[:,0], mesh[:,1], '.')
# plt.show()


# pkl_file = open("C:/Users/Rylei/Documents/SimpleDTQ-LejaMesh.p", "wb" )  
# pickle.dump(mesh, pkl_file)
# pkl_file.close()

pickle_in = open("C:/Users/Rylei/Documents/SimpleDTQ-LejaMesh.p","rb")
mesh = pickle.load(pickle_in)
pdf = UM.generateICPDF(mesh[:,0], mesh[:,1], .1, .1)

# mesh = UM.generateOrderedGridCenteredAtZero(xmin, xmax, xmin, xmax, kstep, includeOrigin=True)

# pdf = UM.generateICPDF(mesh[:,0], mesh[:,1], .1, .1)

# mesh = meshSoln
# pdf = surfaces[15]


# xmin = np.min(mesh[:,0]); xmax = np.max(mesh[:,0])
# ymin = np.min(mesh[:,1]); ymax = np.max(mesh[:,1])

# mesh = UM.generateOrderedGridCenteredAtZero(-1, 1, -1, 1, 0.05, includeOrigin=True)
# mesh = np.vstack((mesh,mesh2))

#x = np.arange(-0.2, 0.2, .01)
#y = np.arange(-0.2, 0.2, .01)
#xx, yy = np.meshgrid(x, y)
#points = np.ones((np.size(xx),2))
#points[:,0]= xx.ravel()
#points[:,1]=yy.ravel()
#tri1 = Delaunay(points).simplices
#
#
#w = np.arange(0,len(x)-1,1)
##w=np.sort(np.concatenate((w,w)))
#cSimplices = []
#for j in range(len(x)-1):
#    for i in w:
#        #cSimplices.append([i+j*len(x),i+1+j*len(x),i+len(x)+j*len(x)])
#        cSimplices.append([i+j*len(x),i+1+j*len(x),i+len(x)+1+j*len(x)])
#
#v = np.arange(len(x),len(x)*2-1,1)
#for j in range(0,len(x)-1):
#    for i in v:
#        cSimplices.append([i+j*len(x),i+1+j*len(x),i-len(x)+j*len(x)])
#                
#
#s = np.asarray(cSimplices, dtype='int32')
#tri = Delaunay(points)
#tri.simplices = s
#mesh = points
# mesh2 = UM.generateRandomPoints(-2,2,-2,2,500)  # unordered mesh

#rad =  np.sqrt(mesh[:,0]**2 + mesh[:,1]**2)
#for i in range(len(rad)-1,-1,-1):
#    if rad[i] > 1.8:
#        mesh = np.delete(mesh,i,0)

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
# mesh = np.vstack((mesh,mesh2))

#mesh = np.vstack((mesh,[0,0]))
#mesh = np.vstack((mesh,[0.05,0.05]))
#mesh = np.vstack((mesh,[-0.05,0.05]))
#mesh = np.vstack((mesh,[0.05,-0.05]))
#mesh = np.vstack((mesh,[-0.05,-0.05]))

#pdf = np.zeros(len(mesh))
#mesh2 = mesh2
#mesh = np.vstack((mesh,mesh2))
# dx = 1*np.ones((1,len(mesh))).T
# dy = 1*np.ones((1,len(mesh))).T
# delta = np.hstack((dx,dy))
# pdf = UM.generateICPDF(mesh[:,0], mesh[:,1], .1, .1)

# pickle_in = open("C:/Users/Rylei/Documents/SimpleDTQ/PickledData/PdfTrajBimodal.p","rb")
# pdf = pickle.load(pickle_in)
# pdf = pdf[-1]
#pdf = phat_rav
#pdf = np.zeros(len(mesh))
#pdf[-1]=10
#pdf[1000]=10


Meshes = []
PdfTraj = []
PdfTraj.append(np.copy(pdf))
Meshes.append(np.copy(mesh))
#
tri = Delaunay(mesh, incremental=True)
#tri.simplices=s
#order2 = []
numSD = 4

#def PrepareVerticesInfo(mesh):
#    for point in trange(len(mesh)):
#        Vertices.append([])
#        VerticesNum.append([])
#        for currGridPoint in range(len(grid)):
#            vertices, indices = UM.getVerticesForPoint([grid[currGridPoint,0], grid[currGridPoint,1]], mesh, tri) # Points that make up triangle
#            Vertices[point].append(np.copy(vertices))
#            VerticesNum[point].append(np.copy(indices))
#    return Vertices, VerticesNum 
SlopesMax = []  
SlopesMin = []
SlopesMean = []  
Slopes = [] 
pdf = np.copy(PdfTraj[-1])
adjustGrid = True
for i in trange(40):
    Slope = MeshUp.getSlopes(mesh, pdf)
    SlopesMean.append(np.mean(Slope))
    SlopesMin.append(np.min(Slope))
    SlopesMax.append(np.max(Slope))
    Slopes.append(Slope)
    if (i >= 0) and adjustGrid:
        assert np.max(PdfTraj[-1] < 10), "PDF Blew up"
        if (i>=0):
            mesh, pdf, tri, addBool,xmin, xmax, ymin, ymax = MeshUp.addPointsToMeshProcedure(mesh, pdf, tri, kstep, h, xmin, xmax, ymin, ymax)
            if (addBool == 1):
                tri = MeshUp.houseKeepingAfterAdjustingMesh(mesh, tri)
            mesh, pdf, remBool = MeshUp.removePointsFromMeshProcedure(mesh, pdf, tri, True)
            if (remBool == 1):
               tri = MeshUp.houseKeepingAfterAdjustingMesh(mesh, tri)
        
    t=0 
    import untitled7 as u13
    if i >0:
        pdfNew = []
        pdf = np.expand_dims(pdf,axis=1)
        for point in range(len(mesh)):
            Px = mesh[point,0]
            Py = mesh[point,1]
            integral, integral2 = u13.getNewPDFVal(Px, Py, mesh, pdf, 50, h)
            pdfNew.append(integral)
        pdf = np.copy(np.asarray(pdfNew))
        pdf = np.squeeze(pdf)
        PdfTraj.append(np.copy(pdf))
        Meshes.append(np.copy(mesh))
        print('Length of mesh = ', len(mesh))
    else:
        # PdfTraj.append(np.copy(pdf))
        # Meshes.append(np.copy(mesh))
        print('Length of mesh = ', len(mesh))


fig = plt.figure()
ax = Axes3D(fig)
index =-1
ax.scatter(Meshes[index][:,0], Meshes[index][:,1], PdfTraj[index], c='r', marker='.')
index = 50
# ax.scatter(mesh[:,0], mesh[:,1], surfaces[index], c='k', marker='.')

#    
#    
def update_graph(num):
    graph.set_data (Meshes[num][:,0], Meshes[num][:,1])
    graph.set_3d_properties(PdfTraj[num])
    title.set_text('3D Test, time={}'.format(num))
    return title, graph


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
title = ax.set_title('3D Test')
    
graph, = ax.plot(Meshes[-1][:,0], Meshes[-1][:,1], PdfTraj[-1], linestyle="", marker="o")
ax.set_zlim(0, np.max(PdfTraj[-1]))
ani = animation.FuncAnimation(fig, update_graph, frames=len(PdfTraj),
                                          interval=500, blit=False)

plt.show()



timestr = time.strftime("%Y%m%d-%H%M%S")
#
# pkl_file = open("C:/Users/Rylei/Documents/SimpleDTQ/PickledData/PdfTrajBimodal-"+ timestr+".p", "wb" ) 
# pkl_file2 = open("C:/Users/Rylei/Documents/SimpleDTQ/PickledData/MeshesBimodal-"+ timestr+".p", "wb" ) 

# pkl_file = open("C:/Users/Rylei/Documents/SimpleDTQ/PickledData/PdfTrajBimodal.p", "wb" ) 
# pkl_file2 = open("C:/Users/Rylei/Documents/SimpleDTQ/PickledData/MeshesBimodal.p", "wb" ) 

# #    
# pickle.dump(PdfTraj, pkl_file)
# pickle.dump(Meshes, pkl_file2)
# pkl_file.close()
# pkl_file2.close()
# #
#pickle_in = open("C:/Users/Rylei/SyderProjects/SimpleDTQGit/PickledData/PDF.p","rb")
#PdfTraj = pickle.load(pickle_in)
#
#pickle_in = open("C:/Users/Rylei/SyderProjects/SimpleDTQGit/PickledData/Meshes.p","rb")
#Meshes = pickle.load(pickle_in)
#
# #
# import pickle
# pickle_in = open("C:/Users/Rylei/SyderProjects/SimpleDTQGit/PickledData/PdfTraj-20200205-191728.p","rb")
# PdfTraj2 = pickle.load(pickle_in)

# pickle_in = open("C:/Users/Rylei/SyderProjects/SimpleDTQGit/PickledData/Meshes-20200205-191728.p","rb")
# Meshes2 = pickle.load(pickle_in)



#gg = 0
##inds = np.asarray(list(range(0, np.size(x1)*np.size(x2))))
##phat_rav = np.ravel(PdfTraj[gg])
#
#
#si = int(np.sqrt(len(Meshes[gg][:,0])))
##inds_unrav = np.unravel_index(inds, (si, si))
#
#pdfgrid = np.zeros((si,si))
#for i in range(si):
#    for j in range(si):
#        pdfgrid[i,j]=PdfTraj[0][i+j]

# pkl_file = open("C:/Users/Rylei/Documents/SimpleDTQ/PickledData/PDFTrajDenseMovingHill.p", "wb" ) 
# pkl_file2 = open("C:/Users/Rylei/Documents/SimpleDTQ/PickledData/MeshDenseMovingHill.p", "wb" ) 

# pkl_file = open("C:/Users/Rylei/Documents/SimpleDTQ/PickledData/PdfTrajV.p", "wb" ) 
# pkl_file2 = open("C:/Users/Rylei/Documents/SimpleDTQ/PickledData/MeshesV.p", "wb" ) 

# #    
# pickle.dump(PdfTraj, pkl_file)
# pickle.dump(Meshes, pkl_file2)
# pkl_file.close()
# pkl_file2.close()

# Generate data...


plt.scatter(Meshes[1][:,0], Meshes[1][:,1],c=np.log(PdfTraj[1]))
plt.show()
