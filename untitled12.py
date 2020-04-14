import numpy as np
from scipy.interpolate import griddata, interp2d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
import GenerateLejaPoints as LP
import UnorderedMesh as UM
from pyopoly1 import opolynd as op
from pyopoly1 import families as f
from indexing import total_degree_indices



L2Errors = []
LinfErrors = []
L1Errors = []
L2wErrors = []
for step in range(10):        
    #compute errors
    l2w = np.sqrt(np.sum(np.abs((a01[step] - a001[step*10]))**2))/len(mesh)
    print(l2w)
    L2wErrors.append(l2w)

    


x = range(len(L2wErrors))
plt.figure()
plt.semilogy(x, np.asarray(L2wErrors), label = 'L2w Error')
plt.legend()

plt.figure()
plt.semilogy(np.abs(a01[1] - a001[10]), label = 'L2w Error')
plt.legend()
    