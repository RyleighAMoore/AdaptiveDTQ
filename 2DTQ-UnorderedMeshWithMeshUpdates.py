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


# define spatial grid
kstep = 0.1
xmin=-2
xmax=2
ymin=-2
ymax=2
h=0.01


poly = HermitePolynomials(rho=0)
d=2
k = 40    
ab = poly.recurrence(k+1)
lambdas = indexing.total_degree_indices(d, k)
poly.lambdas = lambdas

mesh, two = LP.getLejaPoints(230, np.asarray([[0,0]]).T, poly, candidateSampleMesh = [], returnIndices = False)
mesh = LP.mapPointsBack(0, 0, mesh, 0.1, 0.1)


# plt.scatter(mesh[:,0], mesh[:,1])

pdf = UM.generateICPDF(mesh[:,0], mesh[:,1], 0.1, 0.1)

# import pickle
# pkl_file = open("C:/Users/Rylei/Documents/SimpleDTQ/PickledData/PdfTrajLQTwoHillLongFullSplit.p", "rb" ) 
# pkl_file2 = open("C:/Users/Rylei/Documents/SimpleDTQ/PickledData/MeshesLQTwoHillLongFullSplit1.p", "rb" ) 

# PdfTraj = pickle.load(pkl_file)
# Meshes = pickle.load(pkl_file2)


# pkl_file.close()
# pkl_file2.close()

# mesh = Meshes[5]
# pdf = PdfTraj[5]

Meshes = []
PdfTraj = []
PdfTraj.append(np.copy(pdf))
Meshes.append(np.copy(mesh))

tri = Delaunay(mesh, incremental=True)

numSD = 4

SlopesMax = []  
SlopesMin = []
SlopesMean = []  
Slopes = [] 
pdf = np.copy(PdfTraj[-1])
adjustGrid = True
for i in trange(35):
    Slope = MeshUp.getSlopes(mesh, pdf)
    SlopesMean.append(np.mean(Slope))
    SlopesMin.append(np.min(Slope))
    SlopesMax.append(np.max(Slope))
    Slopes.append(Slope)
    if (i >= 0) and adjustGrid:
        assert np.max(PdfTraj[-1] < 10), "PDF Blew up"
        if (i>=0):
            mesh, pdf, tri, addBool = MeshUp.addPointsToMeshProcedure(mesh, pdf, tri, kstep, h, poly)
            if (addBool == 1):
                tri = MeshUp.houseKeepingAfterAdjustingMesh(mesh, tri)
            mesh, pdf, remBool = MeshUp.removePointsFromMeshProcedure(mesh, pdf, tri, True, poly)
            if (remBool == 1):
               tri = MeshUp.houseKeepingAfterAdjustingMesh(mesh, tri)
        
    t=0 
    import LejaQuadrature as LQ
    if i >-1:
        pdfNew = []
        pdf = np.expand_dims(pdf,axis=1)
        Pxs = []
        Pys = []
        print("Stepping Forward....")
        pdf, condnums, meshTemp = LQ.Test_LejaQuadratureLinearizationOnLejaPoints(mesh, pdf, poly,h)
       
        pdf = np.squeeze(pdf)
        PdfTraj.append(np.copy(pdf))
        Meshes.append(np.copy(mesh))
        print('Length of mesh = ', len(mesh))
        fig = plt.figure()
        ax = Axes3D(fig)
        index =-1
        ax.scatter(Meshes[-1][:,0], Meshes[-1][:,1], PdfTraj[-1], c='r', marker='.')
        
    else:
        print('Length of mesh = ', len(mesh))


fig = plt.figure()
ax = Axes3D(fig)
index =5
ax.scatter(Meshes[index][:,0], Meshes[index][:,1], PdfTraj[index], c='r', marker='.')
index = 50
# ax.scatter(mesh[:,0], mesh[:,1], surfaces[index], c='k', marker='.')

def update_graph(num):
    graph.set_data (Meshes[num][:,0], Meshes[num][:,1])
    graph.set_3d_properties(PdfTraj[num])
    title.set_text('3D Test, time={}'.format(num))
    return title, graph


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
title = ax.set_title('3D Test')
    
graph, = ax.plot(Meshes[-1][:,0], Meshes[-1][:,1], PdfTraj[-1], linestyle="", marker="o")
ax.set_zlim(0, np.max(PdfTraj[1]))
ani = animation.FuncAnimation(fig, update_graph, frames=len(PdfTraj),
                                          interval=500, blit=False)

plt.show()



timestr = time.strftime("%Y%m%d-%H%M%S")
#
# pkl_file = open("C:/Users/Rylei/Documents/SimpleDTQ/PickledData/PdfTrajBimodal-"+ timestr+".p", "wb" ) 
# pkl_file2 = open("C:/Users/Rylei/Documents/SimpleDTQ/PickledData/MeshesBimodal-"+ timestr+".p", "wb" ) 

# pkl_file = open("C:/Users/Rylei/Documents/SimpleDTQ/PickledData/PdfTrajLQTwoHillLongFullSplit.p", "wb" ) 
# pkl_file2 = open("C:/Users/Rylei/Documents/SimpleDTQ/PickledData/MeshesLQTwoHillLongFullSplit1.p", "wb" ) 

# #    
# pickle.dump(PdfTraj, pkl_file)
# pickle.dump(Meshes, pkl_file2)
# pkl_file.close()
# pkl_file2.close()

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
