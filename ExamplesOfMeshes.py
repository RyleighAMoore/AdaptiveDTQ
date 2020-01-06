import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import math


circle_x = 0
circle_y = 0
mesh2 = []
angle = 6
for r in range(angle): # angle
    for j in range(20): # radius
        alpha = r * 2 * math.pi /angle
        # calculating coordinates
        x = j * math.cos(alpha) + circle_x
        y = j * math.sin(alpha) + circle_y
        mesh2.append([x, y])
        
mesh2 = np.asarray(mesh2)
mesh2 = points

plt.scatter(mesh2[:,0], mesh2[:,1], marker='.')


tri = Delaunay(mesh2, furthest_site=False, incremental=False, qhull_options='Q8')
plotTri(tri, mesh2)


#ax = plt.axes(projection='3d')
#
## Data for a three-dimensional line
#zline = np.linspace(0, 15, 1000)
#xline = np.sin(zline)
#yline = np.cos(zline)
#ax.plot3D(xline, yline, zline, 'gray')
#
## Data for three-dimensional scattered points
#zdata = 15 * np.random.random(100)
#xdata = np.sin(zdata) 
#ydata = np.cos(zdata)
#
#plt.plot(xline,yline, '.')


# Concave domain

from scipy.spatial import Delaunay
import UnorderedMesh as UM
np.random.seed(0)
x = 3.0 * np.random.rand(2000)
y = 2.0 * np.random.rand(2000) - 1.0
inside = ((x ** 2 + y ** 2 > 1.0) & ((x - 3) ** 2 + y ** 2 > 1.0))
points = np.vstack([x[inside], y[inside]]).T
tri = Delaunay(points)
UM.plotTri(tri, mesh2)


