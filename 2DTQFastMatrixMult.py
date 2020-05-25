# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 12:50:31 2020

@author: Ryleigh
"""

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
import UnorderedMesh as UM
import MeshUpdates2D as MeshUp
sys.path.insert(1, r'C:\Users\Rylei\Documents\SimpleDTQ\pyopoly1')
from Scaling import GaussScale


# T = 0.01  # final time, code computes PDF of X_T
# s = 0.75  # the exponent in the relation k = h^s
# h = 0.01  # temporal step size
# init = 0  # initial condition X_0
# numsteps = int(np.ceil(T / h))

# assert numsteps > 0, 'The variable numsteps must be greater than 0'
h=0.01
s=0.75
kstep = h ** s
kstep = 0.05
xmin=-1.5
xmax=1.5
ymin=-1.5
ymax=1.5


def generateGRow(point, allPoints, kstep, h):
    row = []
    OrderA = []
    for i in range(len(allPoints)):
        val = kstep**2*fun.G(point[0], point[1], allPoints[i,0], allPoints[i,1], h)
        row.append(val)
        OrderA.append([point[0], point[1], allPoints[i,0], allPoints[i,1]])
    return row

mesh = UM.generateOrderedGridCenteredAtZero(xmin, xmax, xmin, xmax, kstep, includeOrigin=True)
mesh2 = np.copy(mesh)
# pdf = UM.generateICPDF(mesh[:,0], mesh[:,1], 0.1, 0.1)
scale = GaussScale(2)
scale.setMu(np.asarray([[0,0]]).T)
scale.setSigma(np.asarray([np.sqrt(h)*fun.g1(),np.sqrt(h)*fun.g2()]))
pdf = fun.Gaussian(scale, mesh)
# 
# for i in range(len(pdf)):
#     pdf[i] = (16*(mesh[i,0]+ mesh[i,1]))**2
# fig = plt.figure()
# ax = Axes3D(fig)
# index =-1
# ax.scatter(mesh[:,0], mesh[:,1], pdf, c='r', marker='.')

GMat = []
for point in trange(len(mesh)):
    gRow = generateGRow([mesh[point,0], mesh[point,1]], mesh, kstep, h)
    GMat.append(np.copy(gRow))


      
surfaces = [] 
surfaces.append(np.copy(pdf))
t=0
while t < 101:
    print(t)
    pdf = np.matmul(GMat, pdf)
    surfaces.append(np.copy(pdf))
    t=t+1
    

fig = plt.figure()
ax = Axes3D(fig)
index =16
ax.scatter(mesh[:,0], mesh[:,1], surfaces[index], c='r', marker='.')
index =16
ax.scatter(Meshes[index][:,0], Meshes[index][:,1], PdfTraj[index], c='k', marker='.')
# ax.scatter(meshVals[:,0], meshVals[:,1], newPDF, c='k', marker='.')

# 

#  
def update_graph(num):
    graph.set_data(mesh[:,0], mesh[:,1])
    graph.set_3d_properties(surfaces[num])
    title.set_text('3D Test, time={}'.format(num))
    return title, graph

meshSoln = np.copy(mesh)
pdfSoln = surfaces.copy()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
title = ax.set_title('3D Test')

graph, = ax.plot(mesh[:,0], mesh[:,1], surfaces[-1], linestyle="", marker="o")
ax.set_zlim(0, np.max(surfaces[10]))
ani = animation.FuncAnimation(fig, update_graph, frames=len(surfaces),
                                         interval=100, blit=False)

plt.show()

pkl_file = open("C:/Users/Rylei/Documents/SimpleDTQ/PickledData/PDFTraj1.p", "wb" ) 
pkl_file2 = open("C:/Users/Rylei/Documents/SimpleDTQ/PickledData/Mesh1.p", "wb" ) 

# import pickle  
# pickle.dump(surfaces, pkl_file)
# pickle.dump(mesh, pkl_file2)
# pkl_file.close()
# pkl_file2.close()



# fig = plt.figure()
# ax = Axes3D(fig)
# index =3
# ax.scatter(mesh[:,0], mesh[:,1], G, c='r', marker='.')