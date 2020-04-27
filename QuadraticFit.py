import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Functions import *
import sys
sys.path.insert(1, r'C:\Users\Rylei\Documents\SimpleDTQ\pyopoly1')
from variableTransformations import *
from Functions import covPart
import math
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
import random
import UnorderedMesh as UM
from scipy.spatial import Delaunay
import MeshUpdates2D as MeshUp
import pickle
import os
import datetime
import time
import GenerateLejaPoints as LP
import pickle
import LejaQuadrature as LQ
import getPCE as PCE
import distanceMetrics as DM
import sys
from families import HermitePolynomials
import indexing
import LejaPoints as LP
import MeshUpdates2D as meshUp
import opoly1d
import opolynd
from Scaling import GaussScale


def fitQuad(Px,Py, mesh, pdf):
    zobs = np.log(pdf)
    zobs = np.squeeze(zobs)
    xy = mesh.T
    x, y = mesh.T
    
    guess = [1, 1, 1, 1, 1, 1]
    pred_params, uncert_cov = opt.curve_fit(quad, xy, zobs, p0=guess)
    
    c = pred_params
    A= np.asarray([[c[0], c[2]],[c[2],c[1]]])
    B=np.expand_dims(np.asarray([c[3], c[4]]),1)
    
    sigma = np.linalg.inv(A)
    
    Lam, U = np.linalg.eigh(A)
    La = np.diag(Lam)
    mu = -1/2 * U @ np.linalg.inv(La) @ (B.T @ U).T

    zpred = quad(xy, *pred_params)
    # print('True parameters: ', true_params)
    # print('Predicted params:', pred_params)
    # print('Residual, RMS(obs - pred):', np.sqrt(np.mean((zobs - zpred)**2)))
    # print(sigma)
    # print(mu)
    # print("sigmas = ", c[0], c[1])
    # print("mus = ", c[2], c[3])
    scaling = GaussScale(2)
    scaling.setMu(np.asarray([[mu[0][0],mu[1][0]]]).T)
    # scaling.setMu(np.asarray([[0,0]]).T)

    # scaling.setMu(np.asarray([[0,0]]).T)
    scaling.setSigma(np.asarray([np.sqrt(sigma[0,0]),np.sqrt(sigma[1,1])]))
    
    gauss = fun.Gaussian(scaling, xy.T)
    mesh = UM.generateOrderedGridCenteredAtZero(-.3, .3, -.3, .3, 0.01, includeOrigin=True)
    gauss2 = fun.Gaussian(scaling, mesh)
        
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(x, y,np.expand_dims(np.exp(zobs),1)/np.expand_dims(gauss,1), c='r', marker='.')
    # ax.scatter(x, y,np.exp(zobs), c='k', marker='.')
    # ax.scatter(mesh[:,0], mesh[:,1], gauss2)
    # plot(xy, zobs, pred_params)
    # plt.show()
    
    
    return scaling, pdf/np.expand_dims(gauss,1)
    

def main():
    a, b, c, d, e, f = 0.1,0.1,0,0,0,0
    true_params = [a, b, c, d, e, f]
    xy, zobs = generate_example_data(20, true_params)
    x, y = xy

    i = zobs.argmax()
    guess = [1, 1, 1, 1, 1, 1]
    pred_params, uncert_cov = opt.curve_fit(quad, xy, zobs, p0=guess)
    c = pred_params
    A= np.asarray([[c[0], c[2]],[c[2],c[1]]])
    B=np.expand_dims(np.asarray([c[3], c[4]]),1)
    
    covar = np.linalg.inv(A)
    
    Lam, U = np.linalg.eigh(A)
    La = np.diag(Lam)
    mu = -1/2 * U @ np.linalg.inv(La) @ (B.T @ U).T

    zpred = quad(xy, *pred_params)
    # print('True parameters: ', true_params)
    # print('Predicted params:', pred_params)
    print('Residual, RMS(obs - pred):', np.sqrt(np.mean((zobs - zpred)**2)))
    print(covar)
    print(mu)
    # print("sigmas = ", c[0], c[1])
    # print("mus = ", c[2], c[3])
    
    scaling = GaussScale(2)
    scaling.setMu(np.asarray([[mu[0][0],mu[1][0]]]).T)
    # scaling.setMu(np.asarray([[0,0]]).T)

    scaling.setSigma(np.asarray([np.sqrt(covar[0,0]),np.sqrt(covar[1,1])]))
    
    gauss = fun.Gaussian(scaling, xy.T)
    mesh = UM.generateOrderedGridCenteredAtZero(-.3, .3, -.3, .3, 0.01, includeOrigin=True)
    gauss2 = fun.Gaussian(scaling, mesh)


    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y,np.exp(zobs)/gauss, c='r', marker='.')
    
    print (np.exp(zobs)/gauss)
    # ax.scatter(mesh[:,0], mesh[:,1], gauss2)
    # plot(xy, zobs, pred_params)
    # plt.show()
    

def quad(xy, a, b, c, d, e, f):
    x, y = xy
    A= np.asarray([[a,c],[c,b]])
    B=np.asarray([[d, e]]).T
    # quad = f + B.T@xy + xy.T@A@xy
    quad = -(a*x**2/2 + b*y**2/2 + 2*c*x*y + d*x + e*y + f)
    # quad = (-(x-c)**2/(2*a) + (y-d)**2/(2*b) + e*x*y +f)
    return quad


def generate_example_data(num, params):
    np.random.seed(1977) # For consistency
    # xy = np.random.random((2, num))
    IC = .1
    H = HermitePolynomials(rho=0)
    d=2
    k = 10  
    ab = H.recurrence(k+1)
    lambdas = indexing.total_degree_indices(d, k)
    H.lambdas = lambdas
    c = H.canonical_connection(len(lambdas))
    xy, two = LP.getLejaPoints(6, np.asarray([[0,0]]).T, H, candidateSampleMesh = [], returnIndices = False)
    xy = LP.mapPointsBack(0, 0, xy, IC, IC)
    
    # xy = UM.generateOrderedGridCenteredAtZero(-1, 1, -1, 1, 0.05, includeOrigin=True)

    xy = xy.T
    x,y = xy
    

    
    # noise = np.log(np.abs(np.random.normal(0, 0.1, num)))
    # zobs2 = quad(xy, *params)
    # zobs2 = np.log(UM.generateICPDF(x, y, .1,.1)) 
    scaling = GaussScale(2)
    scaling.setMu(np.asarray([[0,0]]).T)
    scaling.setSigma(np.asarray([IC,IC]))
    zobs = np.log((UM.generateICPDF(xy.T[:,0], xy.T[:,1], IC,IC))**1)
    # zobs = np.ones(len(xy.T))

    # zobs = np.log(fun.Gaussian(scaling, xy.T)*(2*np.pi*IC*IC))
    # zobs = np.log(np.exp(-((x-.1)**2/(2*IC**2)+ (y-.1)**2/(2*IC**2))))
    # 
    # sigma = 1
    # zobs = -((x-0.5)**2 + y**2)
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(x, y,-zobs, c='r', marker='.')
    # ax.scatter(x, y,zobs2, c='k', marker='.')
    
    return xy, zobs

def plot(xy, zobs, pred_params):
    x, y = xy
    yi, xi = np.mgrid[-2:2:30j, -2:2:30j]
    xyi = np.vstack([xi.ravel(), yi.ravel()])

    zpred = quad(xyi, *pred_params)
    zpred.shape = xi.shape

    fig, ax = plt.subplots()
    ax.scatter(x, y, c=zobs, s=200, vmin=zpred.min(), vmax=zpred.max())
    ax.scatter(x, y, c='k', s=200)

    im = ax.imshow(zpred, extent=[xi.min(), xi.max(), yi.min(), yi.max()],
                   aspect='auto')
    fig.colorbar(im)
    ax.invert_yaxis()
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(xi, yi,zpred, c='r', marker='.')
    ax.scatter(x, y,zobs, c='k', marker='.')

    return fig

# main()
# 