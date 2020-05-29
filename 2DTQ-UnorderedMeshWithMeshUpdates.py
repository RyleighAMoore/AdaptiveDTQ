import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import Functions as fun
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm, trange
import UnorderedMesh as UM
from scipy.spatial import Delaunay
import time
import sys
sys.path.insert(1, r'C:\Users\Rylei\Documents\SimpleDTQ\pyopoly1')
import LejaPoints as LP
import LejaQuadrature as LQ
import distanceMetrics as DM
from families import HermitePolynomials
import indexing
import LejaPoints as LP
import MeshUpdates2D as MeshUp
from Scaling import GaussScale
import ICMeshGenerator as M

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

# pkl_file = open("C:/Users/Rylei/Documents/SimpleDTQ/PickledData/ICMesh1.p", "rb" ) 
# mesh = pickle.load(pkl_file)

mesh = M.getICMesh(1)
pdf = UM.generateICPDF(mesh[:,0], mesh[:,1], IC,IC)

scale = GaussScale(2)
scale.setMu(np.asarray([[0,0]]).T)
scale.setSigma(np.asarray([np.sqrt(h)*fun.g1(),np.sqrt(h)*fun.g2()]))
pdf = fun.Gaussian(scale, mesh)


Meshes = []
PdfTraj = []
PdfTraj.append(np.copy(pdf))
Meshes.append(np.copy(mesh))

tri = Delaunay(mesh, incremental=True)

numSD = 4


pdf = np.copy(PdfTraj[-1])
adjustGrid = True
for i in trange(5):
    if (i >= 2) and adjustGrid:
        assert np.max(PdfTraj[-1] < 10), "PDF Blew up"
        if (i>=0):
            mesh, pdf, tri, addBool = MeshUp.addPointsToMeshProcedure(mesh, pdf, tri, kstep, h, poly)
            if (addBool == 1):
                tri = MeshUp.houseKeepingAfterAdjustingMesh(mesh, tri)
            mesh, pdf, remBool = MeshUp.removePointsFromMeshProcedure(mesh, pdf, tri, True, poly)
            if (remBool == 1):
                tri = MeshUp.houseKeepingAfterAdjustingMesh(mesh, tri)
        
    t=0 
    # print(len(mesh))
    import LejaQuadrature as LQ
    if i >-1:
        pdfNew = []
        pdf = np.expand_dims(pdf,axis=1)
        Pxs = []
        Pys = []
        print("Stepping Forward....")

        pdf, condnums, meshTemp = LQ.Test_LejaQuadratureLinearizationOnLejaPoints(mesh, pdf, poly,h,12, i)

        pdf = np.squeeze(pdf)
        PdfTraj.append(np.copy(pdf))
        Meshes.append(np.copy(mesh))
        print('Length of mesh = ', len(mesh))
        
    else:
        print('Length of mesh = ', len(mesh))


fig = plt.figure()
ax = Axes3D(fig)
index =-1
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
ax.set_zlim(0, np.max(PdfTraj[-1]))
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


# plt.scatter(Meshes[1][:,0], Meshes[1][:,1],c=np.log(PdfTraj[1]))
# plt.show()
