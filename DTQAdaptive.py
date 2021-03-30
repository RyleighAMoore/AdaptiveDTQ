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
from exactSolutions import TwoDdiffusionEquation
from Errors import ErrorValsExact


def DTQ(NumSteps, minSpacing, maxSpacing, h, degree, meshRadius, drift, diff):
    '''Mesh updates parameter'''
    addPointsToBoundaryIfBiggerThanTolerance = 10**(-degree)
    removeZerosValuesIfLessThanTolerance = 10**(-degree-0.5)
    minDistanceBetweenPoints = minSpacing #min(0.12, kstep)
    maxDistanceBetweenPoints = maxSpacing
    conditionNumForAltMethod = 3
    NumLejas = 10
    start = datetime.now()
    
    ''' Initializd orthonormal Polynomial family'''
    poly = HermitePolynomials(rho=0)
    d=2
    k = 40    
    ab = poly.recurrence(k+1)
    lambdas = indexing.total_degree_indices(d, k)
    poly.lambdas = lambdas
    
    '''pdf after one time step with Dirac initial condition centered at the origin'''
    mesh = M.getICMesh(meshRadius, minSpacing, h)

    scale = GaussScale(2)
    scale.setMu(h*drift(np.asarray([0,0])).T)
    scale.setCov((h*diff(np.asarray([0,0]))*diff(np.asarray([0,0])).T).T)
    
    pdf = fun.Gaussian(scale, mesh)
    
    Meshes = []
    PdfTraj = []
    PdfTraj.append(np.copy(pdf))
    Meshes.append(np.copy(mesh))
    
    '''Delaunay triangulation for finding the boundary '''
    tri = Delaunay(mesh, incremental=True)
    
    # needLPBool = numpy.zeros((2, 2), dtype=bool)
    
    '''Initialize Transition probabilities'''
    maxDegFreedom = len(mesh)*2
    # NumLejas = 15
    numQuadFit = max(20,20*np.max(diff(np.asarray([0,0]))).astype(int))*2
    
    GMat = np.empty([maxDegFreedom, maxDegFreedom])*np.NaN
    for i in range(len(mesh)):
        v = fun.G(i,mesh, h, drift, diff)
        GMat[i,:len(v)] = v
        
    LPMat = np.ones([maxDegFreedom, NumLejas])*-1
    LPMatBool = np.zeros((maxDegFreedom,1), dtype=bool) # True if we have Lejas, False if we need Lejas
        
    '''Grid updates'''
    LPReuseArr = []
    Timing = []
    AltMethod = []
    QuadFitRecomputed = []
    Timing.append(start)
    
    
    for i in trange(1,NumSteps+1):
        if (i >= 0):
            '''Add points to mesh'''
            # plt.figure()
            # plt.scatter(mesh[:,0], mesh[:,1])
            mesh, pdf, tri, addBool, GMat = MeshUp.addPointsToMeshProcedure(mesh, pdf, tri, minSpacing, h, poly, GMat, addPointsToBoundaryIfBiggerThanTolerance, removeZerosValuesIfLessThanTolerance, minDistanceBetweenPoints,maxDistanceBetweenPoints, drift, diff)
            # plt.plot(mesh[:,0], mesh[:,1], '*r')

        if i>=15 and i%10==9:
            '''Remove points from mesh'''
            mesh, pdf, GMat, LPMat, LPMatBool, tri = MeshUp.removePointsFromMeshProcedure(mesh, pdf, tri, True, poly, GMat, LPMat, LPMatBool, removeZerosValuesIfLessThanTolerance)
              
        print('Length of mesh = ', len(mesh))
        if i >-1: 
            '''Step forward in time'''
            pdf = np.expand_dims(pdf,axis=1)
            pdf, condnums, meshTemp, LPMat, LPMatBool, LPReuse, AltMethodCount = LQ.Test_LejaQuadratureLinearizationOnLejaPoints(mesh, pdf, poly,h,NumLejas, i, GMat, LPMat, LPMatBool, numQuadFit, removeZerosValuesIfLessThanTolerance, conditionNumForAltMethod, drift, diff)
            pdf = np.squeeze(pdf)
            '''Add new values to lists for graphing'''
            PdfTraj.append(np.copy(pdf))
            Meshes.append(np.copy(mesh))
            LPReuseArr.append(LPReuse)
            time = datetime.now()
            Timing.append(time)
            AltMethod.append(AltMethodCount)
             
        else:
            print('Length of mesh = ', len(mesh))
        
        sizer = len(mesh)
        if np.shape(GMat)[0] - sizer < sizer:
            GMat2 = np.empty([2*sizer, 2*sizer])*np.NaN
            GMat2[:sizer, :sizer]= GMat[:sizer, :sizer]
            GMat = GMat2
                
        if np.shape(LPMat)[0] - sizer < sizer:
            LPMat2 = np.ones([2*sizer, NumLejas])*-1
            LPMat2[:sizer,:]= LPMat[:sizer, :]
            LPMat = LPMat2
            LPMatBool2 = np.zeros((2*sizer,1), dtype=bool)
            LPMatBool2[:len(mesh)]= LPMatBool[:len(mesh)]
            LPMatBool = LPMatBool2
        
    
    end = datetime.now()
    Timing.append(end)
    print("Time: ", end-start)

    surfaces = []
    for ii in range(len(PdfTraj)):
        ana = TwoDdiffusionEquation(Meshes[ii],diff(np.asarray([0,0]))[0,0], h*(ii+1),drift(np.asarray([0,0]))[0,0])
        surfaces.append(ana)

    LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsExact(Meshes, PdfTraj, surfaces, plot=True)
    return Meshes, PdfTraj, LinfErrors, L2Errors, L1Errors, L2wErrors, Timing, LPReuseArr, AltMethod

