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
from Functions import diff

'''Plotting Parameters'''
PlotAnimation = True
PlotFigure = False
PlotStepIndex = -1

'''Initialization Parameters'''
NumSteps = 200
'''Discretization Parameters'''
a = 1
h=0.01
kstep = min(0.15, 0.144*fun.diff(np.asarray([0,0]))[0,0]+0.0056)

'''Errors'''
ComputeErrors = False
# SolutionPDFFile = 'PickledData/SolnPDF-Erf.p'
# SolutionMeshFile = 'PickledData/SolnMesh-Erf.p'
# SolutionPDFFile = 'SolnPDF-ErfIC.p'
# SolutionMeshFile = 'SolnMesh-ErfIC.p'
twiceQuadFit = False
numLejas = 10

Meshes, PdfTraj, LinfErrors, L2Errors, L1Errors, L2wErrors, Timing, LPReuseArr, AltMethod= DTQ(NumSteps, kstep, h, numLejas,twiceQuadFit, 3.5)

x = np.arange(h,(NumSteps+1.5)*h,h)
plt.figure()
plt.semilogy(x, np.asarray(LinfErrors), label = 'Linf Error, deg=2')
plt.semilogy(x, np.asarray(L2Errors), label = 'L2 Error, deg=2')
plt.semilogy(x, np.asarray(L1Errors), label = 'L1 Error, deg=2')
plt.semilogy(x, np.asarray(L2wErrors), label = 'L2w Error, deg=2')
plt.xlabel('Time')
plt.ylabel('Error')
# plt.ylim([10**(-8), 10**(-1)])
plt.legend()


from matplotlib import rcParams
# Font styling
rcParams['font.family'] = 'serif'
rcParams['font.weight'] = 'bold'
rcParams['font.size'] = '12'
fontprops = {'fontweight': 'bold'}
'''Plot figure'''
plt.figure()
# ax = Axes3D(fig)
M= []
S = []
index = -1
# for x in Meshes[0]:
#     M.append(x)
for x in Meshes[index]:
    M.append(x)
# for x in PdfTraj[0]:
#     S.append(x)
for x in PdfTraj[index]:
    S.append(x)
M = np.asarray(M)
S = np.asarray(S)
# plt.scatter(M[:,0], M[:,1], c=S, marker='.')
# cbar = plt.colorbar()
# cbar.set_label("PDF value")
# plt.xlabel(r"$x_1$")
# plt.ylabel(r"$x_2$")

# plt.show()

# plt.tricontour(M[:,0], M[:,1], S, levels=15, linewidths=0.5, colors='k', alpha=0.6)
plt.plot(M[:,0], M[:,1], 'k.', markersize='2')
# cntr2 = plt.tricontourf(M[:,0], M[:,1], S, levels=15, cmap="bone_r", vmin=0.001, vmax = 0.06)
cntr2 = plt.tricontourf(M[:,0], M[:,1], S, levels=15, cmap="bone_r")

cbar = plt.colorbar(cntr2)
cbar.set_label("PDF value")
plt.xlabel("First dimension")
plt.ylabel("Second dimension")
plt.show()



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
                                              interval=300, blit=False)
    plt.show()



    