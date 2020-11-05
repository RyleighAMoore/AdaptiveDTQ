import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import Functions as fun
from mpl_toolkits.mplot3d import Axes3D
from tqdm import trange
from scipy.spatial import Delaunay
import LejaQuadrature as LQ
from pyopoly1.families import HermitePolynomials
from pyopoly1 import indexing
import MeshUpdates2D as MeshUp
from pyopoly1.Scaling import GaussScale
import ICMeshGenerator as M
import pickle  
from Errors import ErrorVals
from datetime import datetime
from pyopoly1 import variableTransformations as VT
import pyopoly1.LejaPoints as LP


start = datetime.now()

'''Plotting Parameters'''
PlotAnimation = True
PlotFigure = False
PlotStepIndex = -1

'''Initialization Parameters'''
NumSteps = 5
adjustBoundary =True
adjustDensity = False # Density changes are not working well right now 
maxDegFreedom = 2000

'''Discretization Parameters'''
kstep = 0.1
h=0.01

'''Errors'''
ComputeErrors = True
# Make sure the file matches the Function.py functions used.
SolutionPDFFile = './PickledData/SolnPDF-Vol.p'
SolutionMeshFile = './PickledData/SolnMesh-Vol.p'
SolutionPDFFile = './PickledData/SolnPDF-Erf.p'
SolutionMeshFile = './PickledData/SolnMesh-Erf.p'

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
scale.setSigma(np.asarray([np.sqrt(h)*fun.diff(np.asarray([[0,0]]))[0,0],np.sqrt(h)*fun.diff(np.asarray([[0,0]]))[1,1]]))
pdf = fun.Gaussian(scale, mesh)


'''Initialize Transition probabilities'''
GMat = np.empty([maxDegFreedom, maxDegFreedom])
for i in range(len(mesh)):
    v = fun.G(i,mesh, h)
    GMat[i,:len(v)] = v
    
'''Iniitialize Leja Point Matrix'''
# LPMatCanonical = []
# LPMatIndices = np.empty([maxDegFreedom, 12])

# Sigma = np.sqrt(h)*fun.diff(mesh) # sigma of gaussian for weight
# mu = h*fun.drift(mesh)
# for ii in range(len(mesh)):
#     muX = mesh[ii,0]  # mean of gaussian for weight
#     muY = mesh[ii,1]   
    
#     scaling = GaussScale(2)
#     scaling.setMu(np.asarray([[muX,muY]]).T)
#     scaling.setSigma(np.diag(Sigma))
    
#     u = VT.map_to_canonical_space(mesh, scaling)
    
#     normScale = GaussScale(2)
#     normScale.setMu(np.asarray([[0,0]]).T)
#     normScale.setCov(np.asarray([[1,0],[0,1]]))
    
#     meshtemp, pdfNew, indices = LP.getLejaSetFromPoints(normScale, u, 12, poly, pdf)
    
#     LPs = []
#     for i in range(len(indices)):
#         LPs.append(mesh[indices[i],:])
        
#     mesh2 = np.asarray(LPs)
#     # mesh2 = VT.map_from_canonical_space(mesh2, scaling)
    
#     LPMatCanonical.append(mesh2)
#     LPMatIndices[ii,:len(indices)] = indices
#     pdf2 = pdf[indices]
    
#     # fig = plt.figure()
#     # ax = Axes3D(fig)
#     # ax.scatter(mesh[:,0], mesh[:,1],pdf , c='k', marker='o')
#     # ax.scatter(mesh2[:,0], mesh2[:,1],pdf2 , c='r', marker='.')
#     # plt.show()
    
    
    

# LPMatIndices = LPMatIndices.astype(int)
Meshes = []
PdfTraj = []
PdfTraj.append(np.copy(pdf))
Meshes.append(np.copy(mesh))

'''Delaunay triangulation for finding the boundary '''
tri = Delaunay(mesh, incremental=True)

'''Grid updates'''
for i in trange(NumSteps):
    # if (i >= 2) and (adjustBoundary or adjustDensity):
    #     '''Add points to mesh'''
    #     mesh, pdf, tri, addBool = MeshUp.addPointsToMeshProcedure(mesh, pdf, tri, kstep, h, poly, adjustBoundary =adjustBoundary, adjustDensity=adjustDensity)
    #     if (addBool == 1): 
    #         '''Recalculate triangulation if mesh was changed'''
    #         tri = MeshUp.houseKeepingAfterAdjustingMesh(mesh, tri)
    #     '''Remove points from mesh'''
    #     mesh, pdf, remBool = MeshUp.removePointsFromMeshProcedure(mesh, pdf, tri, True, poly)
    #     if (remBool == 1): 
    #         '''Recalculate triangulation if mesh was changed'''
    #         tri = MeshUp.houseKeepingAfterAdjustingMesh(mesh, tri)

    print('Length of mesh = ', len(mesh))
    if i >-1: 
        '''Step forward in time'''
        print("Stepping Forward....")
        pdf = np.expand_dims(pdf,axis=1)
        if i==0:
            pdf, condnums, meshTemp, LPMatIndices = LQ.Test_LejaQuadratureLinearizationOnLejaPoints(mesh, pdf, poly,h,12, i, GMat, [], time=True)
        else:
            pdf, condnums, meshTemp, LPMatIndices = LQ.Test_LejaQuadratureLinearizationOnLejaPoints(mesh, pdf, poly,h,12, i, GMat, LPMatIndices, time=False)
        pdf = np.squeeze(pdf)
        '''Add new values to lists for graphing'''
        PdfTraj.append(np.copy(pdf))
        Meshes.append(np.copy(mesh))
     
    # else:
    #     print('Length of mesh = ', len(mesh))

end = now = datetime.now()
print("Time: ", end-start)


'''Plot figure'''
if PlotFigure:
    fig = plt.figure()
    ax = Axes3D(fig)
    index =PlotStepIndex
    ax.scatter(Meshes[index][:,0], Meshes[index][:,1], PdfTraj[index], c='r', marker='.')
    plt.show()

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
    ax.set_zlim(0, np.max(PdfTraj[2]))
    ani = animation.FuncAnimation(fig, update_graph, frames=len(PdfTraj),
                                              interval=500, blit=False)
    plt.show()

'''Errors'''
if ComputeErrors:
    pkl_file = open(SolutionPDFFile, "rb" ) 
    pkl_file2 = open(SolutionMeshFile, "rb" ) 
    mesh2 = pickle.load(pkl_file2)
    surfaces = pickle.load(pkl_file)
    
    ErrorVals(Meshes, PdfTraj, mesh2, surfaces)


# indx = 1218
# indices = LPMatIndices[indx,:] 
# plt.figure()
# plt.plot(mesh[indices,0], mesh[indices,1], 'o')
# plt.plot(mesh[indx,0], mesh[indx,1], '.')
# plt.show()





