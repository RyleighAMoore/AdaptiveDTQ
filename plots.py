# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 18:06:14 2020

@author: Rylei
"""

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

sizes = []
Times = []
T = []
for i in range(1,len(Meshes)):
    sizes.append(len(Meshes[i]))
    Times.append((Timing[i]-Timing[i-1]).total_seconds())
    T.append((Timing[i]-Timing[0]).total_seconds())
    

ii = np.linspace(1,len(PdfTraj)-1,len(PdfTraj)-1)/100
# plt.plot(ii, np.asarray(LPReuseArr),'o', label = 'Reused Leja Points')
plt.plot(ii,sizes,'.',label = 'Mesh Size')
# plt.plot(ii,np.asarray(AltMethod),'o',label = 'Alt. Method Used')
plt.xlabel('Time')
plt.ylabel('Number of Points')
# plt.legend()

fig, axs = plt.subplots(3)
axs[0].plot(ii,sizes,'.')
axs[0].set(ylabel="Number of Points")
axs[1].plot(ii, 100*np.asarray(LPReuseArr)/sizes,'.')
axs[1].set(ylabel="% Reusing Leja Points")
axs[2].plot(ii,100*np.asarray(AltMethod)/sizes,'.')
axs[2].set(xlabel="Time", ylabel="% Using Alt. Method")
# axs[0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))



# plt.legend()

# plt.plot(ii, np.asarray(Times),'o',label = 'Reused Leja Points')

plt.figure()
plt.plot(ii, np.asarray(T)/60,'.')
plt.title("Cumulative Timing vs. Time Step: Erf")
plt.xlabel('Time')
plt.ylabel('Cumulative time in minutes')


plt.figure()
plt.plot(ii, np.asarray(Times)/sizes, '.')
plt.title("Timing vs. Degrees of Freedom")
plt.xlabel('Step Size')
plt.ylabel('Time per Point (seconds)')

plt.plot(sizes, np.asarray(Times)/sizes, '.')





