import DTQAdaptive as D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import Functions as fun
from DriftDiffFunctionBank import MovingHillDrift, DiagDiffptThree
from DTQFastMatrixMult import MatrixMultiplyDTQ
from exactSolutions import TwoDdiffusionEquation
from Errors import ErrorValsExact
from datetime import datetime

mydrift = MovingHillDrift
mydiff = DiagDiffptThree

'''Initialization Parameters'''
NumSteps = 99

x = [1,2,3,4,5,6,7,8,9,10,15]
x = [1,3,5,7]
# x = [7,5,3,1]

# x=[0]
# x=[1,2]
h=0.01
times = np.asarray(np.arange(h,(NumSteps+2)*h,h))


L2wErrorArray = np.zeros((len(x),len(times)))
LengthArray = []

TimingArray = []

a = 1
kstepMin = 0.05 # lambda
kstepMax = 0.07 # Lambda
radius = 0.5 # R
count = 0
mTimes = []
numTimes = 1
for i in x:
    for j in range(numTimes):
        start = datetime.now()
        Meshes, PdfTraj, LPReuseArr, AltMethod= D.DTQ(NumSteps, kstepMin, kstepMax, h, i, radius, mydrift, mydiff, PrintStuff=False)
        end = datetime.now()
        time = end-start
        mTimes.append(time.total_seconds())
    TimingArray.append(sum(mTimes)/numTimes)
    
    surfaces = []
    for ii in range(len(PdfTraj)):
        ana = TwoDdiffusionEquation(Meshes[ii],mydiff(np.asarray([0,0]))[0,0], h*(ii+1), mydrift(np.asarray([0,0]))[0,0])
        surfaces.append(ana)
    
    LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsExact(Meshes, PdfTraj, surfaces, plot=False)

    L2wErrorArray[count,:] = np.asarray(L2wErrors)
    count = count+1
    

'''Discretization Parameters'''

x = [0.1, 0.15, 0.18]
x = [0.06, 0.05, 0.04, 0.03]
# x=[0.1,.15]

h=0.01
times = np.asarray(np.arange(h,(NumSteps+2)*h,h))

L2wErrorArrayT = np.zeros((len(x),len(times)))
timesArrayT = []
stepArrayT = []
LengthArrayT = []

TimingArrayT = []

# xmin = -2.5
# xmax = 5.1
# ymin = -3
# ymax = 3.1

# xmin = -0.5
# xmax = 4.5
# ymin = -1.5
# ymax = 1.5

meshF = np.asarray(Meshes[0])
meshL = np.asarray(Meshes[-1])

'''Find essential support for tensorized version'''
xmin = min(np.min(meshF[:,0]), np.min(meshL[:,0]))
xmax = max(np.max(meshF[:,0]), np.max(meshL[:,0]))
ymin = min(np.min(meshF[:,1]), np.min(meshL[:,1]))
ymax = max(np.max(meshF[:,1]), np.max(meshL[:,1]))
# print("!!!!!!!!!")
count = 0
mTimesT = []
for i in x:
    for j in range(numTimes):
        start = datetime.now()
        mesh, surfaces = MatrixMultiplyDTQ(NumSteps, i, h, mydrift, mydiff, xmin, xmax, ymin, ymax)
        end = datetime.now()
        time = end-start
        mTimesT.append(time.total_seconds())
    TimingArrayT.append(sum(mTimesT)/numTimes)
    
    LengthArrayT.append(len(mesh))
    Meshes = []
    for i in range(len(surfaces)):
        Meshes.append(mesh)
        
    solution = []
    for ii in range(len(surfaces)):
        ana = TwoDdiffusionEquation(Meshes[ii],mydiff(np.asarray([0,0]))[0,0], h*(ii+1),mydrift(np.asarray([0,0]))[0,0])
        solution.append(ana)
    
    LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsExact(Meshes, surfaces, solution, plot=False)
    
    L2wErrorArrayT[count,:] = np.asarray(L2wErrors)
    count = count+1
    
import pickle
with open('L2wErrorArrayT.pickle', 'wb') as f:
    pickle.dump(L2wErrorArrayT, f)
f.close()
with open('L2wErrorArray.pickle', 'wb') as f:
    pickle.dump(L2wErrorArray, f)
f.close()
with open('TimingArrayT.pickle', 'wb') as f:
    pickle.dump(np.asarray(TimingArrayT), f)
f.close()
with open('TimingArray.pickle', 'wb') as f:
    pickle.dump(np.asarray(TimingArray), f)
f.close()
    
mm = min(min(TimingArrayT), min(TimingArray))
mm = TimingArray[0]

from matplotlib import rcParams

# Font styling
rcParams['font.family'] = 'serif'
rcParams['font.weight'] = 'bold'
rcParams['font.size'] = '18'
fontprops = {'fontweight': 'bold'}

m = np.max(np.asarray(TimingArrayT)/mm) 
# nearest_multiple = int(5 * round(m/5))
plt.figure()
# plt.yticks(np.arange(0, nearest_multiple+10, 5))
plt.semilogx(L2wErrorArrayT[:,-1],np.asarray(TimingArrayT)/mm, 'o-',label="Tensorized")
plt.semilogx(L2wErrorArray[:,-1],np.asarray(TimingArray)/mm, 'o:r', label="Adaptive")
plt.semilogx(L2wErrorArray[0,-1],np.asarray(TimingArray[0])/mm, 'ok', label="Unit Time")
plt.ylabel("Relative Time")
plt.xlabel(r"$L_{2w}$ Error")
plt.legend()
plt.show()
    
    
   