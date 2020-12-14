from DTQAdaptive import DTQ
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
from exactSolutions import TwoDdiffusionEquation
from Errors import ErrorValsExact

'''Plotting Parameters'''
PlotAnimation = True
PlotFigure = False
PlotStepIndex = -1

'''Initialization Parameters'''
NumSteps = 100

'''Discretization Parameters'''
a = 1
h=0.01
kstep = 0.2

'''Errors'''
ComputeErrors = False
# Make sure the file matches the Function.py functions used.
# SolutionPDFFile = './PickledData/SolnPDF-Vol.p'
# SolutionMeshFile = './PickledData/SolnMesh-Vol.p'
SolutionPDFFile = 'PickledData/SolnPDF-Erf.p'
SolutionMeshFile = 'PickledData/SolnMesh-Erf.p'
# SolutionPDFFile = 'SolnPDF-ErfIC.p'
# SolutionMeshFile = 'SolnMesh-ErfIC.p'
twiceQuadFit = True
numLejas = 10

# Meshes, PdfTraj, LinfErrors, L2Errors, L1Errors, L2wErrors, Timing, LPReuseArr, AltMethod= DTQ(NumSteps, kstep, h, numLejas,twiceQuadFit, 2)
Meshes2, PdfTraj2, LinfErrors2, L2Errors2, L1Errors2, L2wErrors2, Timing2, LPReuseArr2, AltMethod2= DTQ(NumSteps, kstep, h, numLejas,True, 4)
Meshes3, PdfTraj3, LinfErrors3, L2Errors3, L1Errors3, L2wErrors3, Timing3, LPReuseArr3, AltMethod3= DTQ(NumSteps, kstep, h, numLejas,False, 4)
# Meshes4, PdfTraj4, LinfErrors4, L2Errors4, L1Errors4, L2wErrors4, Timing4, LPReuseArr4, AltMethod4= DTQ(NumSteps, kstep, h, numLejas,twiceQuadFit, 5)
# Meshes5, PdfTraj5, LinfErrors5, L2Errors5, L1Errors5, L2wErrors5, Timing5, LPReuseArr5, AltMethod5= DTQ(NumSteps, kstep, h, numLejas,twiceQuadFit, 6)


x = np.arange(0,(NumSteps+1)*h,h)
plt.figure()
# plt.semilogy(x, np.asarray(LinfErrors), label = 'Linf Error, deg=2')
# plt.semilogy(x, np.asarray(L2Errors), label = 'L2 Error, deg=2')
# plt.semilogy(x, np.asarray(L1Errors), label = 'L1 Error, deg=2')
# plt.semilogy(x, np.asarray(L2wErrors), label = 'L2w Error, deg=2')

plt.semilogy(x, np.asarray(LinfErrors2), '-.', label = 'Twice Quad fit: Linf Error, deg=3')
plt.semilogy(x, np.asarray(L2Errors2),'-.', label = 'Twice Quad fit: L2 Error, deg=3')
plt.semilogy(x, np.asarray(L1Errors2),'-.', label = 'Twice Quad fit: L1 Error, deg=3')
plt.semilogy(x, np.asarray(L2wErrors2),'-.', label = 'Twice Quad fit: L2w Error, deg=3')

plt.semilogy(x, np.asarray(LinfErrors3), ':', label = 'Linf Error, deg=3')
plt.semilogy(x, np.asarray(L2Errors3),':', label = 'L2 Error, deg=3')
plt.semilogy(x, np.asarray(L1Errors3),':', label = 'L1 Error, deg=3')
plt.semilogy(x, np.asarray(L2wErrors3),':', label = 'L2w Error, deg=4')
# plt.semilogy(x, np.asarray(L2wErrors4),':', label = 'L2w Error, deg=5')
# plt.semilogy(x, np.asarray(L2wErrors5),':', label = 'L2w Error, deg=6')

plt.xlabel('Time')
plt.ylabel('Error')
plt.ylim([10**(-8), 10**(-1)])
plt.legend()

'''Plot figure'''
if PlotFigure:
    fig = plt.figure()
    ax = Axes3D(fig)
    index =10
    ax.scatter(Meshes[index][:,0], Meshes[index][:,1], PdfTraj[index], c='r', marker='.')
    plt.show()

# '''Plot Animation'''
if PlotAnimation:
    def update_graph(num):
        graph.set_data (Meshes[num][:,0], Meshes[num][:,1])
        graph.set_3d_properties(PdfTraj[num])
        title.set_text('3D Test, time={}'.format(num))
        return title, graph
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    title = ax.set_title('3D Test')
        
    graph, = ax.plot(Meshes[-1][:,0], Meshes[-1][:,1], PdfTraj[-1], linestyle="", marker=".")
    ax.set_zlim(0, np.max(PdfTraj[-1]))
    mini = np.min(Meshes[0])
    maxi = np.max(Meshes[0])
    for i in range(len(Meshes)):
        m = np.min(Meshes[i])
        M = np.max(Meshes[i])
        if m< mini:
            mini = m
        if M > maxi:
            maxi = M
        
    ax.set_xlim(mini, maxi)
    ax.set_ylim(mini, maxi)
    ani = animation.FuncAnimation(fig, update_graph, frames=len(PdfTraj),
                                              interval=100, blit=False)
    plt.show()


# '''Errors'''
# if ComputeErrors:
#     pkl_file = open(SolutionPDFFile, "rb" ) 
#     pkl_file2 = open(SolutionMeshFile, "rb" ) 
#     mesh2 = pickle.load(pkl_file2)
#     surfaces = pickle.load(pkl_file)
#     ErrorVals(Meshes, PdfTraj, mesh2, surfaces)


# fig = plt.figure()
# ax = Axes3D(fig)
# index =20
# ana = TwoDdiffusionEquation(Meshes[index],0.5, 0.01*(index+1),2)
# ax.scatter(Meshes[index][:,0], Meshes[index][:,1], PdfTraj[index], c='r', marker='.')
# ax.scatter(Meshes[index][:,0], Meshes[index][:,1],ana, c='k', marker='.')
# plt.show()


    