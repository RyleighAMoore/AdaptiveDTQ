import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import math
from scipy.spatial import Delaunay
import UnorderedMesh as UM


# Radial mesh
circle_x = 0
circle_y = 0
mesh2 = []
angle = 50
for r in range(angle): # angle
    for j in range(10): # radius
        alpha = r * 2 * math.pi /angle
        # calculating coordinates
        x = 1/(j+1) * math.cos(alpha) + circle_x
        y = 1/(j+1) * math.sin(alpha) + circle_y
        mesh2.append([x, y])
        
#for r in range(angle): # angle
#    for j in [0.5, 0.4, 0.3, 0.2, 0.1, 0.25, 0.2, 0.15, 0.01]: # radius
#        alpha = r * 2 * math.pi /angle
#        # calculating coordinates
#        x = (j)* math.cos(alpha) + circle_x
#        y = (j) * math.sin(alpha) + circle_y
#        mesh2.append([x, y])

#mesh2.append([0,0])   
  
mesh2 = np.asarray(mesh2)
#grid = UM.makeOrderedGridAroundPoint([0,0],0.01, 100, -0.04,0.04,-0.04,0.04)
#mesh2 = np.vstack((mesh2,grid))

plt.scatter(mesh2[:,0], mesh2[:,1])
tri = Delaunay(mesh2)
UM.plotTri(tri, mesh2)



# Concave domain
#from scipy.spatial import Delaunay
#import UnorderedMesh as UM
#np.random.seed(0)
#x = 3.0 * np.random.rand(2000)
#y = 2.0 * np.random.rand(2000) - 1.0
#inside = ((x ** 2 + y ** 2 > 1.0) & ((x - 3) ** 2 + y ** 2 > 1.0))
#points = np.vstack([x[inside], y[inside]]).T
#tri = Delaunay(points)
#UM.plotTri(tri, mesh2)


x = np.arange(-1, 1, .1)
y = np.arange(-1, 1, .1)
xx, yy = np.meshgrid(x, y)
points = np.ones((np.size(xx),2))
points[:,0]= xx.ravel()
points[:,1]=yy.ravel()
#tri1 = Delaunay(points).simplices


w = np.arange(0,len(x)-1,1)
#w=np.sort(np.concatenate((w,w)))
cSimplices = []
#for j in range(len(x)-1):
#    for i in w:
#        #cSimplices.append([i+j*len(x),i+1+j*len(x),i+len(x)+j*len(x)])
#        cSimplices.append([i+j*len(x),i+1+j*len(x),i+len(x)+1+j*len(x)])

v = np.arange(len(x),len(x)*2-1,1)
for j in range(0,len(x)-1):
    for i in v:
        cSimplices.append([i+j*len(x),i+1+j*len(x),i-len(x)+j*len(x)])
                

s = np.asarray(cSimplices, dtype='int32')
tri = Delaunay(points)
tri.simplices = s
UM.plotTri(tri, points)


    
        






