from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import Functions as fun
import Integrand
import Operations2D
import XGrid
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


T = 0.01  # final time, code computes PDF of X_T
s = 0.75  # the exponent in the relation k = h^s
h = 0.01  # temporal step size
init = 0  # initial condition X_0
numsteps = int(np.ceil(T / h))

assert numsteps > 0, 'The variable numsteps must be greater than 0'

# define spatial grid
kstep = h ** s
kstep = 0.15
# kstep = 0.1
xMin1 = -3
xMax1 = 3
xMin2 = -3.2
xMax2 = 3.2

epsilonTol = -20


x1 = np.arange(xMin1, xMax1, kstep)
x2 = np.arange(xMin2, xMax2, kstep)

#########################################################################################
animate = True
finalGraph = True

# tolerance parameters
epsilonTolerance = -10
minSizeGAndStillRemoveValsFromG = 100
minMaxOfPhatAndStillRemoveValsFromG = 0.001

# simulation parameters
autoCorrectInitialGrid = True
RandomXvec = True  # if autoCorrectInitialGrid is True this has no effect.

RemoveFromG = True  # Also want AddToG to be true if true
IncGridDensity = True
DecGridDensity = True
AddToG = True
###############################################################################################


xList = [] # Create list of xvalues for each slice
for i in range(len(x2)):
    xList.append(np.copy(x1))
    
w1 = Operations2D.find_nearest(x1, 0)
w2 = Operations2D.find_nearest(x2, 0)

X, Y = np.meshgrid(x1, x2) 
phat = np.zeros([len(x1), len(x2)])
a1 = init + fun.f1(init,0)
b1 = np.abs(fun.g1() * np.sqrt(h))
a2 = init + fun.f2(init,0)
b2 = np.abs(fun.g2() * np.sqrt(h))
phat0 = fun.dnorm(x1, a1, b1)  # pdf after one time step with Dirac \delta(x-init)
phat1 = fun.dnorm(x2, a2, b2)  # pdf after one time step with Dirac \delta(x-init)

phat[w1, :] = phat1
phat[:, w2] = phat0

phatList = [] # Create list of xvalues for each slice
for i in range(len(x2)):
    phatList.append(phat[:, i])


xTraj =[]
pdfTraj =[]
xTraj.append(np.copy(xList))
pdfTraj.append(np.copy(phatList))


inds = np.asarray(list(range(0, np.size(x1)*np.size(x2))))
phat_rav = np.ravel(phat)

inds_unrav = np.unravel_index(inds, (len(x1), len(x2)))


Gmat = np.zeros([len(inds_unrav[0]), len(inds_unrav[1])])
for i in range(0, len(inds_unrav[0])): # I
    print(i)
    print(len(inds_unrav[0]))            
    for k in range(0, len(inds_unrav[0])): # K
        Gmat[i,k]=kstep**2*fun.G(x1[inds_unrav[0][i]], x2[inds_unrav[1][i]], x1[inds_unrav[0][k]], x2[inds_unrav[1][k]], h)

t=0
while t < 120:
    print(t)
    phat_rav = np.matmul(Gmat, phat_rav)
    phatMat = np.reshape(phat_rav,(len(x1),len(x2))) 
    t = t+1



def update_graph(num):
    xframe, yframe, zframe = ([],[],[])
    for i in range(len(xTraj[num])): #I changed this to xTraj[num] for futureproofing
        xframe += list(xTraj[num][i])
        yframe += list(np.ones(len(x1))*x2[i])
        zframe += list(pdfTraj[num][i])
    data=df[df['time']==num]
    graph.set_data (xframe, yframe)
    graph.set_3d_properties(zframe)
    title.set_text('3D Test, time={}'.format(num))
    return title, graph, 


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
title = ax.set_title('3D Test')
xframe, yframe, zframe = ([],[],[])

for i in range(len(xTraj[0])):
    xframe += list(xTraj[0][i])
    yframe += list(np.ones(len(x1))*x2[i])
    zframe += list(pdfTraj[0][i])
    
data=df[df['time']==0]
graph, = ax.plot(xframe, yframe, zframe, linestyle="", marker="o")

ani = animation.FuncAnimation(fig, update_graph, frames=len(xTraj),
                                         interval=1000, blit=False)

plt.show()
