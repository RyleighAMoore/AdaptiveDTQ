# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 12:50:31 2020

@author: Ryleigh
"""
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import Functions as fun
import Integrand
import Operations2D
import XGrid
from mpl_toolkits.mplot3d import Axes3D
import QuadRules
from tqdm import tqdm, trange
import UnorderedMesh as UM
import MeshUpdates2D as MeshUp
from pyopoly1.Scaling import GaussScale

from datetime import datetime
from exactSolutions import TwoDdiffusionEquation
from Errors import ErrorValsExact

from DriftDiffFunctionBank import FourHillDrift, DiagDiffptSevenFive, MovingHillDrift, DiagDiffOne

def MatrixMultiplyDTQ(NumSteps, kstep, h, degree, drift, diff, xmin, xmax, ymin, ymax):
    mydrift = drift
    mydiff = diff
    
    # mydrift = MovingHillDrift
    # mydiff = DiagDiffOne
    
    start = datetime.now()
    
    
    # T = 0.01  # final time, code computes PDF of X_T
    # s = 0.75  # the exponent in the relation k = h^s
    # h = 0.01  # temporal step size
    # init = 0  # initial condition X_0
    # numsteps = int(np.ceil(T / h))
    
    # assert numsteps > 0, 'The variable numsteps must be greater than 0'
    # h=0.01
    # s=0.75
    # kstep = h ** s
    # kstep = 0.1
    # xmin=-8
    # xmax=8.1
    # ymin=-8
    # ymax=8.1
    
    # xmin=-2
    # xmax=2
    # ymin=-4
    # ymax=4
    
    
    # def generateGRow(point, allPoints, kstep, h):
    #     row = []
    #     OrderA = []
    #     for i in range(len(allPoints)):
    #         val = kstep**2*fun.G(point[0], point[1], allPoints[i,0], allPoints[i,1], h)
    #         row.append(val)
    #         OrderA.append([point[0], point[1], allPoints[i,0], allPoints[i,1]])
    #     return row
    
    X, Y = np.mgrid[xmin:xmax:kstep, ymin:ymax:kstep]
    mesh = np.vstack([X.ravel(), Y.ravel()]).T
    
    
    # mesh = UM.generateOrderedGridCenteredAtZero(xmin, xmax, ymin, ymax, kstep, includeOrigin=True)
    scale = GaussScale(2)
    # scale.setMu(np.asarray([[0,0]]).T)
    scale.setMu(h*mydrift(np.asarray([0,0])).T)
    scale.setCov((h*mydiff(np.asarray([0,0]))*mydiff(np.asarray([0,0])).T).T)
    pdf = fun.Gaussian(scale, mesh)
    # 
    # for i in range(len(pdf)):
    #     pdf[i] = (16*(mesh[i,0]+ mesh[i,1]))**2
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # index =-1
    # ax.scatter(mesh[:,0], mesh[:,1], pdf, c='r', marker='.')
    
    # GMat = []
    # for point in trange(len(mesh)):
    #     gRow = generateGRow([mesh[point,0], mesh[point,1]], mesh, kstep, h)
    #     GMat.append(np.copy(gRow))
    '''Initialize Transition probabilities'''
    GMat = np.empty([len(mesh), len(mesh)])
    for i in trange(len(mesh)):
        v = kstep**2*fun.G(i,mesh, h, mydrift, mydiff)
        GMat[i,:len(v)] = v
    
          
    surfaces = [] 
    surfaces.append(np.copy(pdf))
    t=0
    while t < 200:
        print(t)
        pdf = np.matmul(GMat, pdf)
        surfaces.append(np.copy(pdf))
        t=t+1
        
    end = datetime.now()
    print("Time: ", end-start)
    
    # ana = TwoDdiffusionEquation(mesh, 1,0.01)
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # index =0
    # ax.scatter(mesh[:,0], mesh[:,1], surfaces[index], c='r', marker='.')
    # ax.scatter(mesh[:,0], mesh[:,1], ana, c='k', marker='.')
    # index =20
    # ax.scatter(Meshes[index][:,0], Meshes[index][:,1], PdfTraj[index], c='k', marker='.')
    # ax.scatter(meshVals[:,0], meshVals[:,1], newPDF, c='k', marker='.')
    
    # 
    
    #  
    def update_graph(num):
        graph.set_data(mesh[:,0], mesh[:,1])
        graph.set_3d_properties(surfaces[num])
        title.set_text('3D Test, time={}'.format(num))
        return title, graph
    
    meshSoln = np.copy(mesh)
    pdfSoln = surfaces.copy()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    title = ax.set_title('3D Test')
    
    graph, = ax.plot(mesh[:,0], mesh[:,1], surfaces[2], linestyle="", marker="o")
    ax.set_zlim(0, 1)
    ani = animation.FuncAnimation(fig, update_graph, frames=len(surfaces),
                                             interval=1000, blit=False)
    
    plt.show()
    
    # import pickle  
    
    # pkl_file0 = open("C:/Users/Rylei/Documents/SimpleDTQ/PickledData/Gpt02Erf.p", "wb" ) 
    # pickle.dump(GMat, pkl_file0)
    # pkl_file0.close()
    
    # pkl_file = open("SolnPDF-ErfIC5.p", "wb" ) 
    # pkl_file2 = open("SolnMesh-ErfIC5.p", "wb" ) 
    
    # import pickle  
    # pickle.dump(surfaces, pkl_file)
    # pickle.dump(mesh, pkl_file2)
    # pkl_file.close()
    # pkl_file2.close()
    
    
    
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # index =3
    # ax.scatter(mesh[:,0], mesh[:,1], G, c='r', marker='.')
    
    # from exactSolutions import TwoDdiffusionEquation
    # from Errors import ErrorValsExact
    # surfaces = []
    # Meshes = []
    # for ii in range(len(pdfSoln)):
    #     ana = TwoDdiffusionEquation(mesh,1, 0.01*(ii+1))
    #     Meshes.append(mesh)
    #     # e = np.max(PdfTraj[ii] - ana)
    #     surfaces.append(ana)
    
    # ErrorValsExact(Meshes, pdfSoln, surfaces)
    
    
    # one = np.zeros(np.shape(GMat))
    # one[0,0] = kstep*0.5
    # one[-1,-1] = kstep*0.5
    
    # two = np.zeros(np.shape(GMat))
    # for i in range(1, len(GMat)-1):
    #     two[i,i] = kstep
    
    # GScale = one+two
    # GmatScaled = GScale@GMat
    
    Meshes=[]
    for i in range(len(surfaces)):
        Meshes.append(mesh)
        
    solution = []
    for ii in range(len(surfaces)):
        ana = TwoDdiffusionEquation(Meshes[ii],mydiff(np.asarray([0,0]))[0,0], h*(ii+1),mydrift(np.asarray([0,0]))[0,0])
        solution.append(ana)
    
    LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsExact(Meshes, surfaces, solution, plot=True)
