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
NumSteps = 35
'''Discretization Parameters'''
a = 1
h=0.01
kstep = np.round(min(0.15, 0.144*fun.diff(np.asarray([0,0]))[0,0]+0.0056),2)

'''Errors'''
ComputeErrors = False
twiceQuadFit = False
numLejas = 10
beta = 10

Meshes, PdfTraj, LinfErrors, L2Errors, L1Errors, L2wErrors, Timing, LPReuseArr, AltMethod= DTQ(NumSteps, kstep, h, numLejas,beta)
# Meshes2, PdfTraj2, LinfErrors2, L2Errors2, L1Errors2, L2wErrors2, Timing2, LPReuseArr2, AltMethod2= DTQ(NumSteps, kstep, h, numLejas,twiceQuadFit, 3.5)

x = np.arange(h,(NumSteps+1.5)*h,h)
plt.figure()
plt.semilogy(x, np.asarray(LinfErrors), 'r', label = 'Linf Error, interp')
plt.semilogy(x, np.asarray(L2Errors), 'b', label = 'L2 Error')
plt.semilogy(x, np.asarray(L1Errors),'g', label = 'L1 Error')
plt.semilogy(x, np.asarray(L2wErrors), 'c', label = 'L2w Error')

# plt.semilogy(x, np.asarray(LinfErrors2), '.r', label = 'Linf Error')
# plt.semilogy(x, np.asarray(L2Errors2), '.b',label = 'L2 Error')
# plt.semilogy(x, np.asarray(L1Errors2), '.g',label = 'L1 Error')
# plt.semilogy(x, np.asarray(L2wErrors2), '.c',label = 'L2w Error')
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
plt.plot(M[:,0], M[:,1], 'k.', markersize='1', alpha=1)
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
    # ax.set_zlim(0, np.max(PdfTraj[10]))
    mini = np.min(Meshes[0])
    maxi = np.max(Meshes[0])
    # for i in range(len(Meshes)):
    #     m = np.min(Meshes[i])
    #     M = np.max(Meshes[i])
    #     if m< mini:
    #         mini = m
    #     if M > maxi:
    #         maxi = M
        
    # ax.set_xlim(mini, maxi)
    # ax.set_ylim(mini, maxi)
    ani = animation.FuncAnimation(fig, update_graph, frames=len(PdfTraj),
                                              interval=300, blit=False)
    plt.show()



# plt.figure()

# ii=69
# plt.scatter(Meshes[ii][:,0], Meshes[ii][:,1])

# ii=68
# plt.scatter(Meshes[ii][:,0], Meshes[ii][:,1])

# nn = []
# mm = []
# for i in range(len(Meshes)):
#     mm.append(len(Meshes[i]))
#     nn.append(len(Meshes2[i]))

# plt.plot(mm)
# plt.plot(nn, '.')
