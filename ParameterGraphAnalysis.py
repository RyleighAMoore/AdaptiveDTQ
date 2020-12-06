import DTQAdaptive as D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

'''Initialization Parameters'''
NumSteps = 25

'''Discretization Parameters'''
kstep = 0.2
h=0.01


LinfErrorArray = []
L1ErrorArray = []
L2wErrorArray = []
x = np.arange(0.1,0.25,0.05)
times = np.asarray(np.arange(0,(NumSteps+1)*h,h))

L2ErrorArray = np.zeros((len(x),len(times)))
timesArray = []
stepArray = []
count = 0
for i in x:
    Meshes, PdfTraj, LinfErrors, L2Errors, L1Errors, L2wErrors, Timing, LPReuseArr, AltMethod= D.DTQ(NumSteps, i, h)
    L2ErrorArray[count,:] = np.asarray(L2Errors)
    # LinfErrorArray.append(LinfErrors)
    # L1ErrorArray.append(L1Errors)
    # L2wErrorArray.append(L2wErrors)
    for j in times:
        timesArray.append(j)
    stepArray.append(i)
    count = count+1
    
    
X,Y = np.meshgrid(times,x)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X, Y, L2ErrorArray, c='r', marker='.')
ax.set_xlabel('time step')
ax.set_ylabel('mesh space')
ax.set_zlabel('Error')
plt.show()    


# plt.figure()
# plt.semilogy(x, np.asarray(LinfErrorArray), label = 'Linf Error')
# plt.semilogy(x, np.asarray(L2Errors), label = 'L2 Error')
# plt.semilogy(x, np.asarray(L1Errors), label = 'L1 Error')
# plt.semilogy(x, np.asarray(L2wErrors), label = 'L2w Error')
# plt.xlabel('Time Step')
# plt.ylabel('Error')
# plt.legend()