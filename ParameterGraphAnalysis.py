import DTQAdaptive as D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import Functions as fun

'''Initialization Parameters'''
NumSteps = 35

'''Discretization Parameters'''
kstep = np.round(min(0.15, 0.144*fun.diff(np.asarray([0,0]))[0,0]+0.0056),2)
h=0.01

x = [1,2,3,4,5,6,7,8,9,10,15,20]
x = [3,4,5,6]

# x=[20]
times = np.asarray(np.arange(h,(NumSteps+2)*h,h))

L2ErrorArray = np.zeros((len(x),len(times)))
LinfErrorArray = np.zeros((len(x),len(times)))
L1ErrorArray = np.zeros((len(x),len(times)))
L2wErrorArray = np.zeros((len(x),len(times)))
timesArray = []
stepArray = []
count = 0
table = ""
for i in x:
    Meshes, PdfTraj, LinfErrors, L2Errors, L1Errors, L2wErrors, Timing, LPReuseArr, AltMethod= D.DTQ(NumSteps, kstep, h, 10, i)
    table = table + str(i) + "&" +str(L2wErrors[-1])+ "&" +str(L2Errors[-1]) + "&" +str(L1Errors[-1]) + "&" +str(LinfErrors[-1])  + "\\\ \hline "
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