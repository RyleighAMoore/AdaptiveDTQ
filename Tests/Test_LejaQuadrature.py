import sys
sys.path.append('C:/Users/Rylei/Documents/SimpleDTQ')

from LejaQuadrature import getLejaQuadratureRule, getNewPDFVal, QuadratureByInterpolation
import numpy as np
from scipy.stats import multivariate_normal
from math import isclose
from scipy.stats import norm
from pyapprox.variables import IndependentMultivariateRandomVariable
from pyapprox.variable_transformations import AffineRandomVariableTransformation
from pyapprox.multivariate_polynomials import PolynomialChaosExpansion, define_poly_options_from_variable_transformation
from pyapprox.indexing import compute_hyperbolic_indices, tensor_product_indices,compute_tensor_product_level_indices
import GenerateLejaPoints as LP
from GenerateLejaPoints import getLejaSetFromPoints, generateLejaMesh, getLejaPoints, mapPointsBack, mapPointsTo
import UnorderedMesh as UM
import numpy as np
import matplotlib.pyplot as plt
from Functions import g1, g2
from mpl_toolkits.mplot3d import Axes3D

def Test_GetLejaQuadratureRule():
    train_samples1, weights1 = getLejaQuadratureRule(0.1, 0.1 ,1,1)
    assert isclose(np.sum(weights1),1)
    train_samples, weights = getLejaQuadratureRule(0.1, 0.1,0,0)
    assert isclose(np.sum(weights),1)
    
    Aa1 = np.matmul(weights1, np.ones(len(train_samples1))) # should be 1
    assert isclose(Aa1[0], 1)
    Aa = np.matmul(weights, np.ones(len(train_samples))) 
    assert isclose(Aa[0], 1)
    
    rv = multivariate_normal([1, -1], [[0.25, 0], [0, 0.25]])
    vals1 = np.asarray([rv.pdf(train_samples1)]).T 
    vals = np.asarray([rv.pdf(train_samples)]).T
    Ac1 = np.matmul(weights1, vals1) # should be 0.000279332
    assert isclose(Ac1[0,0], 0.000279332, abs_tol = 1**(-8))
    Ac = np.matmul(weights, vals) # Should be 0.0130763
    assert isclose(Ac[0,0], 0.0130763, abs_tol = 1**(-8))
    print("GetLejaQuadratureRule - PASS")
    
    var = .01
    sigma=np.sqrt(var)
    train_samples, weights = getLejaQuadratureRule(sigma, sigma ,0,0)
    plt.figure()

    plt.scatter(train_samples[:,0], train_samples[:,1], c='r')
    
    assert isclose(np.sum(weights),1, abs_tol=1**(-5))
    rv = multivariate_normal([0, 0], [[var, 0], [0, var]])
    vals = np.asarray([rv.pdf(train_samples)]).T
    # vals =np.ones(len(vals))

    Ac11 = np.matmul(weights, vals) 
    print(Ac11)

import distanceMetrics as DM  
def Test_LejaQuadratureOnLejaPoints():
    muX = .2
    muY = 0
    h = 0.01
    var = .01
    sigma=np.sqrt(var)
    
    mesh = LP.generateLejaMesh(200, .1, .1, 30)
    print(DM.fillDistance(mesh))
    # plt.figure()

    # plt.scatter(mesh[:,0], mesh[:,1], c='r')
    pdf = UM.generateICPDF(mesh[:,0], mesh[:,1], sigma, sigma)
    value = QuadratureByInterpolation(mesh, pdf, sigma, sigma, muX, muY, 15)
    
    print(value)
    
# Test_GetLejaQuadratureRule()
Test_LejaQuadratureOnLejaPoints()