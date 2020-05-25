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


# d = 2
# k = 150
# H = f.HermitePolynomials()
# ab = H.recurrence(k+1)

# lambdas = total_degree_indices(d, 100)
# H.lambdas = lambdas

# # V = op.opolynd_eval(Meshes[0], lambdas[:len(PdfTraj[0]),], ab, H)
# V = op.opolynd_eval(mesh2, lambdas[:len(pdf),], ab, H)

# # c = np.dot(np.linalg.inv(V), PdfTraj[0])
# c = np.dot(np.linalg.inv(V),pdf)


# interp_eval = np.dot(V, c)

# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(mesh2[:,0], mesh2[:,1], interp_eval, c='k', marker='.')
# ax.scatter(mesh2[:,0], mesh2[:,1], pdf, c='k', marker='.')

# # fig = plt.figure()
# # ax = Axes3D(fig)
# # ax.scatter(Meshes[0][:,0], Meshes[0][:,1], interp_eval, c='k', marker='.')
# # ax.scatter(Meshes[0][:,0], Meshes[0][:,1], PdfTraj[0], c='r', marker='.')




L2Errors = []
LinfErrors = []
L1Errors = []
L2wErrors = []
for step in range(len(PdfTraj)):
    # Interpolate the fine grid soln to the leja points
    gridSolnOnLejas = griddata(mesh2, surfaces[1*step], Meshes[step], method='cubic', fill_value=np.min(surfaces[1*step]))
        
    #compute errors
    l2w = np.sqrt(np.sum(np.abs((gridSolnOnLejas - PdfTraj[step]))**2*gridSolnOnLejas)/np.sum(gridSolnOnLejas))
    print(l2w)
    L2wErrors.append(l2w)
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(Meshes[step][:,0], Meshes[step][:,1], (PdfTraj[step]-gridSolnOnLejas), c='k', marker='.')
    maxx = np.argmax(PdfTraj[step])
    ax.scatter(Meshes[step][maxx,0], Meshes[step][maxx,1],0, c='r', marker='o')
    
    l2 = np.sqrt(np.sum(np.abs((gridSolnOnLejas - PdfTraj[step])*1)**2)/len(PdfTraj[step]))
    L2Errors.append(l2)
    
    l1 = np.sum(np.abs(gridSolnOnLejas - PdfTraj[step])*gridSolnOnLejas)/len(PdfTraj[step])
    L1Errors.append(l1)
    
    linf = np.max(np.abs(gridSolnOnLejas - PdfTraj[step]))
    LinfErrors.append(linf)
    

# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(Meshes[1][:,0], Meshes[1][:,1], np.abs((gridSolnOnLejas - PdfTraj[1])), c='k', marker='.')
# ax.scatter(Meshes[1][:,0], Meshes[1][:,1], PdfTraj[0], c='r', marker='.')
# ax.scatter(mesh2[:,0], mesh2[:,1], surfaces[0], c='k', marker='.')


x = range(len(L2Errors))
plt.figure()
plt.semilogy(x, np.asarray(LinfErrors), label = 'Linf Error')
plt.semilogy(x, np.asarray(L2Errors), label = 'L2 Error')
plt.semilogy(x, np.asarray(L1Errors), label = 'L1 Error')
plt.semilogy(x, np.asarray(L2wErrors), label = 'L2w Error')
plt.xlabel('Time Step')
plt.ylabel('Error')
plt.legend()

diffs = []
for step in range(len(PdfTraj)):
    err = np.abs(surfaces[step][0] - PdfTraj[step][0])
    diffs.append(err)
    
t = np.asarray(diffs)
    
plt.figure()
plt.semilogy(range(len(PdfTraj)), diffs)
plt.show()
    

idx = 14
m = max(np.round(PdfTraj[idx],5))
maxVals = [i for i, j in enumerate(np.round(PdfTraj[idx],5)) if j == m]  

plt.figure()
x,y = Meshes[idx].T
for val in maxVals:
    plt.scatter(x.T[val],y.T[val])
    