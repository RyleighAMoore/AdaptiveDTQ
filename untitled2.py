import DTQAdaptive as D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import Functions as fun
from DriftDiffFunctionBank import FourHillDrift, DiagDiffptSevenFive
from DTQFastMatrixMult import MatrixMultiplyDTQ

from datetime import datetime

mydrift = FourHillDrift
mydiff = DiagDiffptSevenFive


'''Initialization Parameters'''
NumSteps = 199

'''Discretization Parameters'''


x = [0.1, 0.15, 0.18]
x = [0.04,0.05, 0.06]

x=[0.12]
h=0.01
times = np.asarray(np.arange(h,(NumSteps+2)*h,h))

L2ErrorArray = np.zeros((len(x),len(times)))
LinfErrorArray = np.zeros((len(x),len(times)))
L1ErrorArray = np.zeros((len(x),len(times)))
L2wErrorArray = np.zeros((len(x),len(times)))
timesArray = []
stepArray = []
count = 0
table = ""

a = 1
#kstepMin = np.round(min(0.15, 0.144*mydiff(np.asarray([0,0]))[0,0]+0.0056),2)
# kstep = 0.12 # lambda
xmin = -5.2
xmax = 5.2
ymin = -5.2
ymax = 5.2

# xmin = -1.5
# xmax = 1.5
# ymin = -1.5
# ymax = 1.5

for i in x:
    start = datetime.now()
    Meshes, PdfTraj, LinfErrors, L2Errors, L1Errors, L2wErrors = MatrixMultiplyDTQ(NumSteps, i, h, mydrift, mydiff, xmin, xmax, ymin, ymax)
    end = datetime.now()
    time = end-start
    
    table = table + str(i) + "&" +str("{:2e}".format(L2wErrors[-1]))+ "&" +str("{:2e}".format(L2Errors[-1])) + "&" +str("{:2e}".format(L1Errors[-1])) + "&" +str("{:2e}".format(LinfErrors[-1]))  + "&" + str(len(Meshes)) + "&" + str(time)+ "\\\ \hline "
    L2ErrorArray[count,:] = np.asarray(L2Errors)
    LinfErrorArray[count,:] = np.asarray(LinfErrors)
    L1ErrorArray[count,:] = np.asarray(L1Errors)
    L2wErrorArray[count,:] = np.asarray(L2wErrors)
    for j in times:
        timesArray.append(j)
    stepArray.append(i)
    count = count+1
    
    
X,Y = np.meshgrid(times,x)
fig = plt.figure()
ax = Axes3D(fig)
# ax.scatter(X, Y, L2wErrorArray, c='r', marker='.')
ax.scatter(X, Y, np.log(L2wErrorArray), c='r', marker='.')

ax.set_xlabel('time step')
ax.set_ylabel('mesh space')
ax.set_zlabel('Error')
plt.show()    

from matplotlib import rcParams
# Font styling
rcParams['font.family'] = 'serif'
rcParams['font.weight'] = 'bold'
rcParams['font.size'] = '12'
fontprops = {'fontweight': 'bold'}

plt.figure()
count = 0
for k in x:
    print(count)
    # plt.semilogy(x, LinfErrorArray[k,:], label = 'Linf Error')
    # plt.semilogy(x, L2Errors[k,:], label = 'L2 Error')
    # plt.semilogy(x, np.asarray(L1Errors), label = 'L1 Error')
    plt.semilogy(times, L2wErrorArray[count,:], label = r'$\beta = %d$' %stepArray[count])
    plt.xlabel('Time')
    plt.ylabel(r'$L_{2w}$ Error')
    plt.legend()
    count = count+1
    
    
mesh = Meshes
surfaces = PdfTraj
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

graph, = ax.plot(mesh[:,0], mesh[:,1], surfaces[2], linestyle="", marker="o")
ax.set_zlim(0, 1)
ani = animation.FuncAnimation(fig, update_graph, frames=len(surfaces),
                                         interval=20, blit=False)

plt.show()