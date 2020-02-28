import numpy as np
from scipy.interpolate import griddata, interp2d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
import GenerateLejaPoints as LP
import UnorderedMesh as UM



def getErrors(meshSol, valSoln, meshTest, valTest):
    # degree=25
    # num_leja_samples = 200
    # num_vars = 2
    maxErrors = []
    L2Errors = []
    location = []
    for i in range(min(len(valSoln), len(meshTest))):
    # for i in range(1):
        for ii in range(1,10):
            xmin = np.min(meshSol[:,0]); xmax = np.max(meshSol[:,0])
            ymin = np.min(meshSol[:,1]); ymax = np.max(meshSol[:,1])
            # xmin = -2; xmax = 2
            # ymin = -2; ymax = 2 
            grid_x, grid_y = np.mgrid[xmin:xmax:0.01*ii, ymin:ymax:0.01*ii]
            location.append(np.asarray([1+i,0.01*ii]))

            # grid_x, grid_y = np.meshgrid(meshTest[i][:,0],meshTest[i][:,1])
    
            # rv = multivariate_normal([0, 0], [[.1, 0], [0, .1]])
            # initial_samples = np.asarray([[0],[0]])
            # train_samples, newLeja = LP.getLejaPoints(num_leja_samples, initial_samples,degree, num_candidate_samples = 5000, dimensions=num_vars)
            
            # train_values = np.asarray([rv.pdf(train_samples)]).T
            
            # soln_mesh = UM.generateOrderedGridCenteredAtZero(-2, 2, -2, 2, 0.01)
            
            # soln_vals = np.asarray([rv.pdf(soln_mesh)]).T
        
            
            grid_train = griddata(meshTest[i], valTest[i], (grid_x, grid_y), method='cubic')
            grid_soln = griddata(meshSol, valSoln[i], (grid_x, grid_y), method='cubic')
            grid_train = np.matrix.flatten(grid_train)
            grid_soln=  np.matrix.flatten(grid_soln)
            assert len(grid_soln) == len(grid_train)
           
            
            gT = np.isnan(grid_train)
            gS = np.isnan(grid_soln)
            grid_train[gT] = 0
            grid_soln[gS] = 0
            maxError = np.max(np.abs(grid_soln - grid_train))
            maxErrors.append(maxError)
            # grid_soln = np.ma.masked_where(grid_soln <= 10**(-10), grid_soln)
            # grid_train = np.ma.masked_where(grid_train <= 10**(-10), grid_train)
            runningsum = 0
            counter = 0
            for y in range(len(grid_soln)):
                # print(len(grid_soln))
                # print(len(grid_train))
                if grid_train[y] > 10**(-8) and grid_soln[y] > 10**(-8):
                    runningsum= runningsum+ np.abs(grid_soln[y] - grid_train[y])**2
                    counter =counter + 1
            L2 = np.sqrt(runningsum) / counter 
            # print(L2)
            L2Errors.append(L2)

            # L2 = np.sqrt(np.sum(np.abs(grid_soln - grid_train)**2))
            # L2Errors.append(L2/len(grid_train))
            # print(len(grid_train))
            
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(grid_x,grid_y, grid_train, c='r', marker='.')
        ax.scatter(grid_x,grid_y, grid_soln, c='k', marker='.')
        plt.show()
            
        # # fig = plt.figure()
        # # ax = Axes3D(fig)
        # # ax.scatter(grid_x,grid_y, grid_soln, c='k', marker='.')
        # # plt.show()
            
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # ax.scatter(grid_x,grid_y, np.abs(grid_soln-grid_train), c='g', marker='.')
        # plt.show()
        
    return maxErrors, L2Errors, grid_soln, grid_train, np.asarray(location)
        
maxError, L2Error,grid_soln, grid_train, locations = getErrors(meshSoln, pdfSoln, Meshes, PdfTraj)

fig = plt.figure()
ax = Axes3D(fig)
# ax.scatter(locations[:,0], locations[:,1], maxError, c='r', label = "Max Error")
ax.scatter(locations[:,0], locations[:,1], L2Error, c='g', label = "L2 Error")

ax.set_xlabel('Time Step', labelpad=20)
ax.set_ylabel('Grid Spacing', labelpad=20)
ax.legend()


fig = plt.figure()
ax = Axes3D(fig)
index =2
ax.scatter(meshSoln[:,0], meshSoln[:,1], pdfSoln[index], c='k', marker='.')
ax.scatter(Meshes[index][:,0], Meshes[index][:,1], PdfTraj[index], c='r', marker='.')

 



Meshes = []
PdfTraj = [] 
pdfSoln = []
# for i in range(2):
for j in range(1,8):
    degree=55
    num_leja_samples = 400
    num_vars = 2
    xmin = -2; xmax = 2
    ymin =-2; ymax = 2

    initial_samples = np.asarray([[0],[0]])
    # train_samples, newLeja = LP.getLejaPoints(num_leja_samples, initial_samples,degree, num_candidate_samples = 5000, dimensions=num_vars)
    train_samples =LP.generateLejaMesh(num_leja_samples, 0.1*j, 0.1*j, degree)
    pdf = UM.generateICPDF(train_samples[:,0], train_samples[:,1], 0.1*j, 0.1*j)
    Meshes.append(train_samples)
    PdfTraj.append(pdf)
    meshSoln = UM.generateOrderedGridCenteredAtZero(xmin, xmax, xmin, xmax, 0.05, includeOrigin=True)
    pdf2 = UM.generateICPDF(meshSoln[:,0], meshSoln[:,1], 0.1*j, 0.1*j)
    pdfSoln.append(pdf2)

     
    

Meshes = []
PdfTraj = [] 
pdfSoln = []
# for i in range(2):
for j in range(1,8):
    degree=55
    num_leja_samples = 100*j
    num_vars = 2
    xmin = -2; xmax = 2
    ymin =-2; ymax = 2

    initial_samples = np.asarray([[0],[0]])
    # train_samples, newLeja = LP.getLejaPoints(num_leja_samples, initial_samples,degree, num_candidate_samples = 5000, dimensions=num_vars)
    train_samples =LP.generateLejaMesh(num_leja_samples, 0.1, 0.1, degree)
    pdf = UM.generateICPDF(train_samples[:,0], train_samples[:,1], 0.1, 0.1)
    Meshes.append(train_samples)
    PdfTraj.append(pdf)
    meshSoln = UM.generateOrderedGridCenteredAtZero(xmin, xmax, xmin, xmax, 0.05, includeOrigin=True)
    pdf2 = UM.generateICPDF(meshSoln[:,0], meshSoln[:,1], 0.1, 0.1)
    pdfSoln.append(pdf2)






# maxErrors = []
# L2Errors = []
# for i in range(1):
#     degree=55
#     num_leja_samples = 100*(i+1)
#     num_vars = 2
#     xmin = -2; xmax = 2
#     ymin =-2; ymax = 2
#     grid_x, grid_y = np.mgrid[xmin:xmax:50j, ymin:ymax:50j]
#     x1 = np.matrix.flatten(grid_x)
#     x2 = np.matrix.flatten(grid_y)
#     mesh = np.vstack((x1,x2)).T
    
#     rv = multivariate_normal([0, 0], [[.1, 0], [0, .1]])
#     initial_samples = np.asarray([[0],[0]])
#     train_samples, newLeja = LP.getLejaPoints(num_leja_samples, initial_samples,degree, num_candidate_samples = 5000, dimensions=num_vars)
    
#     train_values = np.asarray([rv.pdf(train_samples)]).T
    
#     soln_mesh = UM.generateOrderedGridCenteredAtZero(-2, 2, -2, 2, 0.01)
    
#     soln_vals = np.asarray([rv.pdf(mesh)]).T
    
    
#     grid_train = griddata(meshTest[i], valTest[i], (grid_x, grid_y), method='cubic')
#     grid_soln = griddata(meshSol, valSoln[i], (grid_x, grid_y), method='cubic')
#     grid_train = np.matrix.flatten(grid_train)
#             grid_soln=  np.matrix.flatten(grid_soln)
#     #grid_train = griddata(train_samples, train_values, (grid_x, grid_y), method='cubic')

#     # grid_soln = griddata(meshSol, valSoln[i], (grid_x, grid_y), method='cubic')
#     # grid_train = np.matrix.flatten(grid_train)
#     # grid_soln=  np.matrix.flatten(grid_soln)
       
    
#     # gT = np.isnan(grid_train)
#     # gS = np.isnan(grid_soln)
#     # grid_train[gT] = 0
#     # grid_soln[gS] = 0
#     # maxError = np.max(np.abs(soln_vals - grid_train))
#     # maxErrors.append(maxError)
#     # L2 = np.sqrt(np.sum(np.abs(soln_vals - grid_train)**2))
#     # L2Errors.append(L2)
    
#     fig = plt.figure()
#     ax = Axes3D(fig)
#     ax.scatter(x1,x2, soln_vals, c='r', marker='.')
#     plt.show()
    
#     fig = plt.figure()
#     ax = Axes3D(fig)
#     ax.scatter(mesh, grid_train, c='k', marker='.')
#     plt.show()
    
#     # fig = plt.figure()
#     # ax = Axes3D(fig)
#     # ax.scatter(grid_x,grid_y, np.abs(grid_soln-grid_train), c='g', marker='.')
#     # plt.show()
# # plt.figure()
# # plt.semilogy(maxErrors, '.k', label = 'Max Error')
# # plt.semilogy(L2Errors, 'k', label = 'L2 Error')
# # plt.legend()
# # plt.show()
