# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 11:21:05 2020

@author: Ryleigh
"""
import pyapprox

from functools import partial
from pyapprox.polynomial_sampling import *
from pyapprox.multivariate_polynomials import PolynomialChaosExpansion, \
    define_poly_options_from_variable_transformation
from pyapprox.variable_transformations import \
     define_iid_random_variable_transformation, RosenblattTransformation, \
     TransformationComposition 
from pyapprox.indexing import compute_hyperbolic_indices, get_total_degree
from pyapprox.univariate_quadrature import gauss_jacobi_pts_wts_1D,\
    clenshaw_curtis_pts_wts_1D
from pyapprox.models.genz import GenzFunction
from scipy.stats import beta as beta, uniform, norm
from pyapprox.density import tensor_product_pdf
#from pyapprox.configure_plots import *
from pyapprox.utilities import get_tensor_product_quadrature_rule
from pyapprox.tests.test_rosenblatt_transformation import rosenblatt_example_2d
import matplotlib.pyplot as plt
import UnorderedMesh as UM
import numpy as np
import Functions  as fun
plt.rcParams['text.usetex'] = False
import matplotlib.animation as animation
import pickle


def getLejaPoints(num_leja_samples, initial_samples,numBasis, num_candidate_samples = 5000, dimensions=2):
    num_vars=2
    degree=numBasis
    
    poly = PolynomialChaosExpansion()
    var_trans = define_iid_random_variable_transformation(
        norm(loc=0,scale=1),num_vars)
    
    opts = define_poly_options_from_variable_transformation(var_trans)
    poly.configure(opts)
    indices = compute_hyperbolic_indices(num_vars,degree,1.0)
#    indices = get_total_degree(num_vars, num_points)

    poly.set_indices(indices)
    
    degree = np.sqrt(2*num_leja_samples)
    generate_candidate_samples = lambda n: np.sqrt(2*degree)*np.random.normal(0, 1, (num_vars, n))
    candidate_samples = generate_candidate_samples(num_candidate_samples)  
    
#    plt.scatter(candidate_samples[0,:], candidate_samples[1,:])
#    plt.show()
    # enforcing lu interpolation to interpolate a set of initial points
    # before selecting best samples from candidates can cause ill conditioning
    # to avoid this issue build a leja sequence and use this as initial
    # samples and then recompute sequence with different candidates
    
    # must use canonical_basis_matrix to generate basis matrix
    #if initial_samples == None:
    #num_initial_samples = 10
    #else:
    PxInit = initial_samples[0,0]
    PyInit = initial_samples[1,0]
    
#    initial_samples = var_trans.map_to_canonical_space(initial_samples)
#    Px = initial_samples[0,0]
#    Py= initial_samples[1,0]
#    initial_samples = mapPointsTo(Px, Py, initial_samples.T).T
    
    
    num_initial_samples = len(initial_samples.T)
    precond_func = lambda matrix, samples: christoffel_weights(matrix)
#    initial_samples, data_structures = get_lu_leja_samples(
#        poly.canonical_basis_matrix,generate_candidate_samples,
#        num_candidate_samples,num_initial_samples,
#        preconditioning_function=precond_func,
#        initial_samples=initial_samples)
    
    samples, data_structures, successBool = get_lu_leja_samples(
        poly.canonical_basis_matrix,
        candidate_samples,num_leja_samples,
        preconditioning_function=precond_func,
        initial_samples=initial_samples)
    
    if successBool ==True:
#        samples = var_trans.map_from_canonical_space(samples)
#        samples = mapPointsBack(PxInit, PyInit, samples.T)
    #    plot = True
    #    if plot: 
    #        plt.plot(samples.T[:,0], samples.T[:,1], '*r')
    #        plt.plot(initial_samples.T[:,0], initial_samples.T[:,1], '.k')
    #        plt.show()
        return np.asarray(samples).T, np.asarray(samples[:,num_initial_samples:]).T
  
    if successBool == False:
        numInitialAdded = 0
        pointsRemoved = []
        initial_samples_edited = np.copy(initial_samples)
        newLejaSamples = []
        ii=0
        while successBool == False: # Truncate initalSamples until succed to add a Leja point
            print("Truncating Initial Samples")
            assert len(pointsRemoved) <= num_initial_samples, "Removed all Initial points"
            pointsRemoved.append(np.asarray([initial_samples_edited[:,0]]).T)
            initial_samples_edited = np.delete(initial_samples_edited,0,1)
            num_initial_samples_edited = len(initial_samples_edited.T) 
            samples2, data_structures2, successBool = get_lu_leja_samples(poly.canonical_basis_matrix,candidate_samples,num_leja_samples,preconditioning_function=precond_func,initial_samples=initial_samples_edited)
            ii+=1
        initial_samples_edited = np.copy(samples2[:, 0:num_initial_samples_edited+1])
        numInitialAdded = num_initial_samples - ii# Able to add a Leja point!
        newLejaSamples.append(np.copy(initial_samples_edited[:,-1]))
        ii=0
        while len(pointsRemoved) != 0: #Try to add one more point in Leja sequence
            print("Trying to Add a point")
            ii+=1
            pointToAdd = pointsRemoved.pop(-1)
            initial_samples_edited = np.hstack((pointToAdd,initial_samples_edited))
            num_initial_samples_edited = len(initial_samples_edited[1,:])
            num_leja_samples_edited = len(initial_samples_edited[1,:]) # Want to try and add the points
            num_leja_samples_edited = num_leja_samples # Want to try and add the points

            samples2, data_structures2, successBool = get_lu_leja_samples(poly.canonical_basis_matrix,candidate_samples,num_leja_samples_edited,preconditioning_function=precond_func,initial_samples=initial_samples_edited)
            if successBool == True:
#                initial_samples_edited = np.copy(samples2[:,0:num_initial_samples_edited])
                numInitialAdded += 1
                print("successfully Added a Point")
            if successBool == False: 
                pointsRemoved.append(np.asarray([initial_samples_edited[:,0]]).T)
                initial_samples_edited = np.delete(initial_samples_edited,0,1)
                num_leja_samples_edited = len(initial_samples_edited[1,:])+1  # Want to add one leja point
                samples, data_structures, successBool = get_lu_leja_samples(poly.canonical_basis_matrix,candidate_samples,num_leja_samples_edited,preconditioning_function=precond_func,initial_samples=initial_samples_edited)                
                initial_samples_edited = np.copy(samples[0:len(initial_samples_edited[1,:])+1,:])
                newLejaSamples.append(np.copy(initial_samples_edited[:,-1]))
        
#        while len(newLejaSamples) < num_leja_samples-num_initial_samples:
#            num_leja_samples_edited = len(initial_samples_edited[1,:]) +1  # Want to add one leja point
#            samples, data_structures, successBool = get_lu_leja_samples(poly.canonical_basis_matrix,candidate_samples,num_leja_samples_edited,preconditioning_function=precond_func,initial_samples=initial_samples_edited)                
#            initial_samples_edited = np.copy(samples[0:len(initial_samples_edited[1,:])+1,:])
#            newLejaSamples.append(np.copy([initial_samples_edited[:,-1]]))
        for i in range(num_initial_samples_edited, len(samples2[1, :])):    
            newLejaSamples.append(np.asarray(samples2[:, i]))
    samples = samples2[:, :num_leja_samples]
#    samples = var_trans.map_from_canonical_space(samples)
#    samples = mapPointsBack(PxInit, PyInit, samples.T)


#    plot = True
#    if plot: 
#        plt.plot(samples.T[:,0], samples.T[:,1], '*r')
#        plt.plot(initial_samples.T[:,0], initial_samples.T[:,1], '.k')
#        plt.show()    
    return np.asarray(samples).T, np.asarray(newLejaSamples)



#initial = np.asarray([[0,0], [-.2,0], [0,.2]]).T
#lejaPoints = getLejaPoints(8, initial)

#neighbors = UM.findNearestKPoints(Px, Py, allPoints, 4)
"""
point 1x2 array
allPoints nx2 array of the original point and the neighbors we consider.
returns transformed points so that point is centered at 0,0
"""
def mapPointsTo(Px, Py, allPoints,scaleX, scaleY):    
    dx = Px*np.ones((1,len(allPoints))).T
    dy = Py*np.ones((1,len(allPoints))).T
    delta = np.hstack((dx,dy))
    scaleX = scaleX*np.ones((1,len(allPoints))).T
    scaleY = scaleY*np.ones((1,len(allPoints))).T
    scaleVec = np.hstack((scaleX,scaleY))
    return (np.asarray(allPoints) - delta)*scaleVec

def mapPointsBack(Px, Py, allPoints, scaleX, scaleY):    
    dx = Px*np.ones((1,len(allPoints))).T
    dy = Py*np.ones((1,len(allPoints))).T
    delta = np.hstack((dx,dy))
    scaleX = scaleX*np.ones((1,len(allPoints))).T
    scaleY = scaleY*np.ones((1,len(allPoints))).T
    scaleVec = np.hstack((scaleX,scaleY))
    return (scaleVec)*np.asarray(allPoints) + delta

def getLejaPointsWithStartingPoints(Px, Py, numNeighbors, mesh, numNewLejaPoints, scaleX, scaleY, numBasis, numSamples):
    neighbors, distances = UM.findNearestKPoints(Px, Py, mesh, numNeighbors) 
    if len(neighbors > 0): 
        neighbors = np.vstack((neighbors,[Px,Py]))
    else: # make sure we have at least one point.
        neighbors = np.asarray([[Px],[Py]]).T
        
    intialPoints = mapPointsTo(Px,Py,neighbors, 1/scaleX,1/scaleY)
    lejaPointsFinal, newLeja = getLejaPoints(numNewLejaPoints+numNeighbors+1, intialPoints.T, numBasis,num_candidate_samples=numSamples)
    lejaPointsFinal = mapPointsBack(Px,Py,lejaPointsFinal, scaleX, scaleY)
    newLeja = mapPointsBack(Px,Py,newLeja,scaleX,scaleY)
    plot= False
    if plot:
        plt.figure()
        plt.plot(neighbors[:,0], neighbors[:,1], '*k', label='Neighbors', markersize=14)
        plt.plot(Px, Py, '*r',label='Main Point',markersize=14)
        plt.plot(lejaPointsFinal[:,0], lejaPointsFinal[:,1], '.c', label='Leja Points',markersize=10)
        plt.legend()
        plt.show()
    lejaPointsFinal
    return lejaPointsFinal, newLeja


def generateLejaMesh(numPoints, sigmaX, sigmaY, numBasis):
#    lejaPoints, newPoints = getLejaPointsWithStartingPoints(0, 0, 0,[], numPoints, np.sqrt(h)*fun.g1(),np.sqrt(h)*fun.g2(),40)
     lejaPoints, newPoints = getLejaPointsWithStartingPoints(0, 0, 0,[], numPoints, sigmaX,sigmaY,numBasis, 10000)

     return lejaPoints


        
# mesh = UM.generateOrderedGridCenteredAtZero(-1.8, 1.8, -1.8, 1.8, 0.1)      # ordered mesh 
# # mesh = UM.generateRandomPoints(-0.2,0.2,-0.2,0.2,200)  # unordered mesh

# num = 0
# point = np.asarray(mesh[num:num+1,:])
# Px = point[0,0]
# Py= point[0,1]
pickle_in = open("C:/Users/Rylei/Documents/SimpleDTQ-LejaMesh.p","rb")
mesh = pickle.load(pickle_in)
for val in range(len(mesh)-1,-1,-1):
    xx = mesh[val,0]
    yy = mesh[val,1]
    rad = xx**2 +yy**2 
    if rad < 0.1:
        mesh = np.delete(mesh, val, 0)
        
        
        
# diff = np.ones((len(mesh),2))
# diff[:,0]= 3*np.ones((len(mesh)))
# diff[:,1]= 1*np.ones((len(mesh)))
# mesh = np.vstack((mesh,mesh+ diff))

# # lejaPointsFinal, newLeja = getLejaPointsWithStartingPoints(0, 0,0, mesh, 100, 1, 1, 15, 1000)
# lejaPointsFinal, newLeja = getLejaPoints(len(mesh), mesh.T, 55, num_candidate_samples = 5000, dimensions=2)
# def update_graph(num):
#     graph.set_data(lejaPointsFinal[0:num,0], lejaPointsFinal[0:num,1])
#     # graph.set_3d_properties(PdfTraj[num])
#     # title.set_text('3D Test, time={}'.format(num))
#     return graph


# fig = plt.figure()
# ax = fig.add_subplot()
# # title = ax.set_title('3D Test')
    
# graph, = ax.plot(lejaPointsFinal[0,0], lejaPointsFinal[0,1], linestyle="", marker="o")
# ax.set_xlim(-5, 5)
# ax.set_ylim(-5, 5)

# ani = animation.FuncAnimation(fig, update_graph, frames=len(lejaPointsFinal),
#                                           interval=100, blit=False)

# plt.show()

# lejaPoints = generateLejaMesh(10)




