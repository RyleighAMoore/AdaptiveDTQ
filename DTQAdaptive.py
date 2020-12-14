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


def DTQ(NumSteps, kstep, h, NumLejas, twiceQuadFit, degree):
    '''Mesh updates parameter'''
    # degree=3.5
    deg =degree
    addPointsToBoundaryIfBiggerThanTolerance = 10**(-deg)
    removeZerosValuesIfLessThanTolerance = 10**(-deg-0.5)
    minDistanceBetweenPoints = max(0.15, kstep)
    maxDistanceBetweenPoints =  max(0.2, kstep)
    start = datetime.now()
    
    ''' Initializd orthonormal Polynomial family'''
    poly = HermitePolynomials(rho=0)
    d=2
    k = 40    
    ab = poly.recurrence(k+1)
    lambdas = indexing.total_degree_indices(d, k)
    poly.lambdas = lambdas
    
    '''pdf after one time step with Dirac initial condition centered at the origin'''
    mesh = M.getICMesh(1, kstep, h)
    scale = GaussScale(2)
    scale.setMu(h*fun.drift(np.asarray([0,0])).T)
    scale.setCov(h*fun.diff(np.asarray([0,0])).T)
    
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
    # numQuadFit = max(int(np.ceil(4/kstep)),20)
    numQuadFit = 20
    
    GMat = np.empty([maxDegFreedom, maxDegFreedom])*np.NaN
    for i in range(len(mesh)):
        v = fun.G(i,mesh, h)
        GMat[i,:len(v)] = v
        
    LPMat = np.empty([maxDegFreedom, NumLejas])
    LPMatBool = np.zeros((maxDegFreedom,1), dtype=bool) # True if we have Lejas, False if we need Lejas
    
    QuadFitMat = np.empty([maxDegFreedom, numQuadFit])
    QuadFitBool = np.zeros((maxDegFreedom,1), dtype=bool) # True if have points, false if need points
    
    '''Grid updates'''
    LPReuseArr = []
    Timing = []
    AltMethod = []
    QuadFitRecomputed = []
    Timing.append(start)
    
    
    for i in trange(1,NumSteps+1):
        t = NumSteps*kstep
        if (i >= 1):
            '''Add points to mesh'''
            mesh, pdf, tri, addBool, GMat = MeshUp.addPointsToMeshProcedure(mesh, pdf, tri, kstep, h, poly, GMat, addPointsToBoundaryIfBiggerThanTolerance, removeZerosValuesIfLessThanTolerance, minDistanceBetweenPoints,maxDistanceBetweenPoints)
            if i%20==0:
                '''Remove points from mesh'''
                mesh, pdf, GMat, LPMat, LPMatBool, QuadFitBool, QuadFitMat, tri = MeshUp.removePointsFromMeshProcedure(mesh, pdf, tri, True, poly, GMat, LPMat, LPMatBool,QuadFitBool,QuadFitMat, removeZerosValuesIfLessThanTolerance)
              
        print('Length of mesh = ', len(mesh))
        if i >-1: 
            '''Step forward in time'''
            pdf = np.expand_dims(pdf,axis=1)
            pdf, condnums, meshTemp, LPMat, LPMatBool, QuadFitMat,QuadFitBool, LPReuse, AltMethodCount = LQ.Test_LejaQuadratureLinearizationOnLejaPoints(mesh, pdf, poly,h,NumLejas, i, GMat, LPMat, LPMatBool, QuadFitMat,QuadFitBool, numQuadFit, twiceQuadFit)
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
            GMat2 = np.empty([3*sizer, 3*sizer])*np.NaN
            GMat2[:sizer, :sizer]= GMat[:sizer, :sizer]
            GMat = GMat2
                
        if np.shape(LPMat)[0] - sizer < sizer:
            LPMat2 = np.empty([3*sizer, NumLejas])
            LPMat2[:sizer,:]= LPMat[:sizer, :]
            LPMat = LPMat2
            LPMatBool2 = np.zeros((3*sizer,1), dtype=bool)
            LPMatBool2[:len(mesh)]= LPMatBool[:len(mesh)]
            LPMatBool = LPMatBool2
            
        if np.shape(QuadFitBool)[0] - sizer < sizer:
            QuadFitMat2 = np.empty([3*sizer, numQuadFit])
            QuadFitMat2[:sizer,:]= QuadFitMat[:sizer, :]
            QuadFitMat = QuadFitMat2
            QuadFitBool2 = np.zeros((3*sizer,1), dtype=bool)
            QuadFitBool2[:len(mesh)]= QuadFitBool[:len(mesh)]
            QuadFitBool = QuadFitBool2
    
    
    end = datetime.now()
    Timing.append(end)
    print("Time: ", end-start)

    surfaces = []
    for ii in range(len(PdfTraj)):
        ana = TwoDdiffusionEquation(Meshes[ii],1, h*(ii+1),2)
        surfaces.append(ana)

    LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsExact(Meshes, PdfTraj, surfaces, plot=True)
    return Meshes, PdfTraj, LinfErrors, L2Errors, L1Errors, L2wErrors, Timing, LPReuseArr, AltMethod

