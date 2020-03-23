import sys
sys.path.append('C:/Users/Rylei/Documents/SimpleDTQ')

from GenerateLejaPoints import getLejaSetFromPoints, generateLejaMesh, getLejaPoints, mapPointsBack, mapPointsTo
import UnorderedMesh as UM
import numpy as np
import matplotlib.pyplot as plt
from Functions import g1, g2
from mpl_toolkits.mplot3d import Axes3D


def Test_GetLejaPoints1():
     mesh = generateLejaMesh(120, 1, 1, 30)
     plt.figure()
     plt.scatter(mesh[:,0], mesh[:,1])
     plt.show()

def Test_GetLejaSetFromPoints():
    Px = 0
    Py = 0
    h=0.01
    sigmaX =0.1# np.sqrt(h)*g1()
    sigmaY = 0.1#np.sqrt(h)*g2()
    # mesh = UM.generateRandomPoints(-4*sigmaX,4*sigmaX,-4*sigmaY,4*sigmaY,500)  # unordered mesh
    mesh = generateLejaMesh(530, 0.1, 0.1, 40)
    meshtwo = generateLejaMesh(230, 0.2,0.2, 20)

    mesh = np.vstack((mesh,meshtwo))
    mesh = mapPointsBack(Px, Py, mesh, 1,1)
    meshtwo = mapPointsBack(Px, Py, meshtwo, 1,1) # for graphing purposes only


    mesh1 = mapPointsTo(Px, Py, mesh, 1/sigmaX, 1/sigmaY)
    lejas, indices = getLejaSetFromPoints(0, 0, mesh1, 100, 20)
    
    lejas = mapPointsBack(Px, Py, lejas, sigmaX, sigmaY)
    plt.figure()
    plt.scatter(mesh[:,0], mesh[:,1])
    plt.scatter(meshtwo[:,0], meshtwo[:,1], c='k')

    plt.scatter(lejas[:,0], lejas[:,1], c='r', marker='*')
    plt.show()
    pdf = UM.generateICPDF(lejas[:,0], lejas[:,1], .1, .1)
    
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(lejas[:,0], lejas[:,1], pdf, c='r', marker='.')
    
import time  
def Test_GetLejaSetFromPoints2():
    Px = 0
    Py = 0
    h=0.01
    sigmaX =0.1# np.sqrt(h)*g1()
    sigmaY = 0.1#np.sqrt(h)*g2()
    # mesh = UM.generateRandomPoints(-4*sigmaX,4*sigmaX,-4*sigmaY,4*sigmaY,500)  # unordered mesh
    mesh = generateLejaMesh(230, sigmaX, sigmaY, 50)
    # meshtwo = generateLejaMesh(230, 0.2,0.2, 20)

    # mesh = np.vstack((mesh,meshtwo))
    mesh = mapPointsBack(Px, Py, mesh, 1,1)
    # meshtwo = mapPointsBack(Px, Py, meshtwo, 1,1) # for graphing purposes only
    

    # mesh1 = mapPointsTo(Px, Py, mesh, 1/sigmaX, 1/sigmaY)
    start_time = time.time()    
    lejas, indices = getLejaSetFromPoints(0, 0, mesh, 131, 20, sigmaX, sigmaY)
    end_time = time.time()
    print("--- %s seconds ---" % (time.time() - start_time))

    
    # lejas = mapPointsBack(Px, Py, lejas, sigmaX, sigmaY)
    plt.figure()
    plt.scatter(mesh[:,0], mesh[:,1])
    # plt.scatter(meshtwo[:,0], meshtwo[:,1], c='k')

    plt.scatter(lejas[:,0], lejas[:,1], c='r', marker='*')
    plt.show()
    pdf = UM.generateICPDF(lejas[:,0], lejas[:,1], .1, .1)
    
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(lejas[:,0], lejas[:,1], pdf, c='r', marker='.')

    

def Test_GetLejaPoints():
    num_leja_samples = 12
    numBasis = 40
    initial_samples = np.asarray([[0],[0]])
    lejaPoints, newLejas = getLejaPoints(num_leja_samples, initial_samples,numBasis, num_candidate_samples = 5000, dimensions=2, defCandidateSamples=False)
    
    plt.figure()
    plt.scatter(lejaPoints[:,0], lejaPoints[:,1] , c='g')
    plt.show()
    
Test_GetLejaSetFromPoints2()



# Test_GetLejaPoints()
# Test_GetLejaPoints1()