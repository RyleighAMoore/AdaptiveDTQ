import DTQAdaptive as D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

'''Initialization Parameters'''
NumSteps = 20

'''Discretization Parameters'''
kstep = 0.2
h=0.01



x = np.arange(0.05,0.3,0.05)
times = np.asarray(np.arange(0,(NumSteps+1)*h,h))

L2ErrorArray = np.zeros((len(x),len(times)))
LinfErrorArray = np.zeros((len(x),len(times)))
L1ErrorArray = np.zeros((len(x),len(times)))
L2wErrorArray = np.zeros((len(x),len(times)))
timesArray = []
stepArray = []
count = 0
for i in x:
    Meshes, PdfTraj, LinfErrors, L2Errors, L1Errors, L2wErrors, Timing, LPReuseArr, AltMethod= D.DTQ(NumSteps, i, h, 10, True, 4)
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


plt.figure()
for k in range(0, 11,1):
    # plt.semilogy(x, LinfErrorArray[k,:], label = 'Linf Error')
    # plt.semilogy(x, L2Errors[k,:], label = 'L2 Error')
    # plt.semilogy(x, np.asarray(L1Errors), label = 'L1 Error')
    plt.semilogy(times, LinfErrorArray[k,:], label = 'Spatial Step Size %f' %stepArray[k])
    plt.xlabel('Time Step')
    plt.ylabel('Error')
    plt.legend()