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
for step in range(len(Meshes)):
    # Interpolate the fine grid soln to the leja points
    gridSolnOnLejas = griddata(mesh2, surfaces[10*step], Meshes[step], method='cubic', fill_value=0)
        
    #compute errors
    l2w = np.sqrt(np.sum(np.abs((gridSolnOnLejas - PdfTraj[step]))**2))/len(PdfTraj)
    print(l2w)
    L2Errors.append(l2w)
    
    l1 = np.sum(np.abs(gridSolnOnLejas - PdfTraj[step])*gridSolnOnLejas)/len(PdfTraj)
    L1Errors.append(l1)
    
    linf = np.sum(np.abs(gridSolnOnLejas - PdfTraj[step]))
    LinfErrors.append(linf)
    

# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(Meshes[0][:,0], Meshes[0][:,1], gridSolnOnLejas, c='k', marker='.')
# ax.scatter(Meshes[0][:,0], Meshes[0][:,1], PdfTraj[0], c='r', marker='.')
# # ax.scatter(mesh2[:,0], mesh2[:,1], surfaces[0], c='k', marker='.')


x = range(len(PdfTraj))
plt.figure()
plt.semilogy(x, np.asarray(LinfErrors))
plt.semilogy(x, np.asarray(L2Errors))
plt.semilogy(x, np.asarray(L1Errors))

diffs = []
for step in range(len(PdfTraj)):
    err = np.abs(surfaces[step][0] - PdfTraj[step][0])
    diffs.append(err)
    
t = np.asarray(diffs)
    
plt.figure()
plt.semilogy(range(len(PdfTraj)), diffs)
plt.show()
    
    