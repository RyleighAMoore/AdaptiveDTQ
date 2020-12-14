import DTQAdaptive as D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

'''Initialization Parameters'''
NumSteps = 100

'''Discretization Parameters'''
kstep = 0.2
h=0.01


x = [6,10,15]
times = np.asarray(np.arange(0,(NumSteps+1)*h,h))
L2ErrorArray = np.zeros((len(x),len(times)))
LinfErrorArray = np.zeros((len(x),len(times)))
L1ErrorArray = np.zeros((len(x),len(times)))
L2wErrorArray = np.zeros((len(x),len(times)))
timesArray = []
stepArray = []
count = 0
for i in x:
    Meshes, PdfTraj, LinfErrors, L2Errors, L1Errors, L2wErrors, Timing, LPReuseArr, AltMethod= D.DTQ(NumSteps, kstep, h, i)
    L2ErrorArray[count,:] = np.asarray(L2Errors)
    LinfErrorArray[count,:] = np.asarray(LinfErrors)
    L1ErrorArray[count,:] = np.asarray(L1Errors)
    L2wErrorArray[count,:] = np.asarray(L2wErrors)
    count = count+1
    
    
# X,Y = np.meshgrid(times,x)
# fig = plt.figure()
# ax = Axes3D(fig)
# # ax.scatter(X, Y, L2wErrorArray, c='r', marker='.')
# ax.scatter(X, Y, np.log(L2wErrorArray), c='r', marker='.')

# ax.set_xlabel('time step')
# ax.set_ylabel('mesh space')
# ax.set_zlabel('Error')
# plt.show()    


plt.figure()
for k in range(0, len(x),1):
    # plt.semilogy(x, LinfErrorArray[k,:], label = 'Linf Error')
    # plt.semilogy(x, L2Errors[k,:], label = 'L2 Error')
    # plt.semilogy(x, np.asarray(L1Errors), label = 'L1 Error')
    plt.semilogy(times, LinfErrorArray[k,:], label = '# Leja Points %d' %x[k])
    plt.xlabel('Time Step')
    plt.ylabel('Error')
    plt.legend()