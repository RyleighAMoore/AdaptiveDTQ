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

'''Plotting Parameters'''
PlotAnimation = True
PlotFigure = False
PlotStepIndex = -1

'''Initialization Parameters'''
adjustBoundary =True
adjustDensity = False # Density changes are not working well right now 

'''Discretization Parameters'''
kstep = 0.1
h=0.01

''' Initializd orthonormal Polynomial family'''
poly = HermitePolynomials(rho=0)
d=2
k = 40    
ab = poly.recurrence(k+1)
lambdas = indexing.total_degree_indices(d, k)
poly.lambdas = lambdas

'''pdf after one time step with Dirac initial condition centered at the origin'''
mesh = M.getICMesh(1, kstep, h)
scale = GaussScale(2)
scale.setMu(np.asarray([[0,0]]).T)
scale.setSigma(np.asarray([np.sqrt(h)*fun.g1(),np.sqrt(h)*fun.g2()]))
pdf = fun.Gaussian(scale, mesh)


Meshes = []
PdfTraj = []
PdfTraj.append(np.copy(pdf))
Meshes.append(np.copy(mesh))

'''Delaunay triangulation for finding the boundary '''
tri = Delaunay(mesh, incremental=True)

'''Grid updates'''
for i in trange(20):
    if (i >= 2) and (adjustBoundary or adjustDensity):
        '''Add points to mesh'''
        mesh, pdf, tri, addBool = MeshUp.addPointsToMeshProcedure(mesh, pdf, tri, kstep, h, poly, adjustBoundary =adjustBoundary, adjustDensity=adjustDensity)
        if (addBool == 1): 
            '''Recalculate triangulation if mesh was changed'''
            tri = MeshUp.houseKeepingAfterAdjustingMesh(mesh, tri)
        '''Remove points from mesh'''
        mesh, pdf, remBool = MeshUp.removePointsFromMeshProcedure(mesh, pdf, tri, True, poly)
        if (remBool == 1): 
            '''Recalculate triangulation if mesh was changed'''
            tri = MeshUp.houseKeepingAfterAdjustingMesh(mesh, tri)

    print('Length of mesh = ', len(mesh))
    if i >-1: 
        '''Step forward in time'''
        print("Stepping Forward....")
        pdf = np.expand_dims(pdf,axis=1)
        pdf, condnums, meshTemp = LQ.Test_LejaQuadratureLinearizationOnLejaPoints(mesh, pdf, poly,h,12, i)
        pdf = np.squeeze(pdf)
        '''Add new values to lists for graphing'''
        PdfTraj.append(np.copy(pdf))
        Meshes.append(np.copy(mesh))
         
    else:
        print('Length of mesh = ', len(mesh))

'''Plot figure'''
if PlotFigure:
    fig = plt.figure()
    ax = Axes3D(fig)
    index =PlotStepIndex
    ax.scatter(Meshes[index][:,0], Meshes[index][:,1], PdfTraj[index], c='r', marker='.')

'''Plot Animation'''
if PlotAnimation:
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



