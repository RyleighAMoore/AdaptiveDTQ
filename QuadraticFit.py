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


def fitQuad(Px, Py, mesh, pdf):
    h=0.01
    if np.min(pdf) <= 0:
        rrrr=0
    zobs = np.log(pdf)
    zobs = np.squeeze(zobs)
    xy = mesh.T
    x, y = mesh.T
    
    guess = [1, 1, 1, 1, 1, 1]
    try:
        pred_params, uncert_cov = opt.curve_fit(quad, xy, zobs, p0 = [0,0,0,0,0,0])
    except:
        return float('nan'),float('nan'),float('nan'),float('nan')
    
        print(zobs)
        print(mesh)
        print(pdf)
    
    c = pred_params
    A = np.asarray([[c[0], c[2]],[c[2],c[1]]])
    B = np.expand_dims(np.asarray([c[3], c[4]]),1)
    
    # if c[0]*c[1] - c[2]**2 <0:
    #     print(Px,Py)
    # assert c[0]*c[1] - c[2]**2 >0
    ''' 
    if np.linalg.det(A)<=0:
        # print(Px,Py)
        # print(pdf)
        # sigma = np.asarray([[h*fun.g1()**2,0],[0,h*fun.g2()**2]])
        evals, vects = np.linalg.eig(A)
        projSum = np.zeros(np.shape(vects))
        for i in range(len(vects)):
            v = vects[:,i]
            s = max(evals[i],0)*v@v.T
            A = projSum +s
        A[0,1] = A[0,1]/2
        A[1,0] = A[1,0]/2
   
    if A[0,0] <= 0:
        A[0,0] = 0.01
    if A[1,1] <= 0:
        A[1,1] = 0.01
    
    if np.linalg.det(A) <=0:
        A[0,1] = 0
        A[1,0] = 0
        print(A)
    '''
    if np.linalg.det(A)<= 0:
         return float('nan'),float('nan'),float('nan'),float('nan')
        
        # assert c[0] >0
        # assert c[1]>0
        
    sigma = np.linalg.inv(A)

    Lam, U = np.linalg.eigh(A)
    if np.min(Lam) <= 0:
        return float('nan'),float('nan'),float('nan'),float('nan')
    
    La = np.diag(Lam)
    mu = -1/2*U @ np.linalg.inv(La) @ (B.T @ U).T
    # print(sigma)

    L = np.linalg.cholesky((sigma))
    
    Const = np.exp(-c[5]+1/4*B.T@U@np.linalg.inv(La)@U.T@B)
    

    # Const = 1/(np.pi*np.linalg.det(np.abs(L)))
    
    zpred = quad(xy, *pred_params)
    # print('True parameters: ', true_params)
    # print('Predicted params:', pred_params)
    # print('Residual, RMS(obs - pred):', np.sqrt(np.mean((zobs - zpred)**2)))
    # print(sigma)
    # print(mu)
    # print("sigmas = ", c[0], c[1])
    # print("mus = ", c[2], c[3])
    import math
    
    if math.isfinite(mu[0][0]) and math.isfinite(mu[1][0]) and math.isfinite(np.sqrt(sigma[0,0])) and math.isfinite(np.sqrt(sigma[1,1])):

        scaling = GaussScale(2)
        scaling.setMu(np.asarray([[mu[0][0],mu[1][0]]]).T)
    
        # scaling.setMu(np.asarray([[0,0]]).T)
    
        # scaling.setMu(np.asarray([[0,0]]).T)
        # scaling.setSigma(np.asarray([np.sqrt(sigma[0,0]),np.sqrt(sigma[1,1])]))
        scaling.setCov(sigma)
        gauss = fun.Gaussian(scaling, xy.T)
        # mesh = UM.generateOrderedGridCenteredAtZero(-.3, .3, -.3, .3, 0.01, includeOrigin=True)
        # gauss2 = fun.Gaussian(scaling, mesh)
    
    else: 
        stop = 0
        
    cc=pred_params
    x,y = xy   
    vals = np.exp(-(cc[0]*x**2+ cc[1]*y**2 + 2*cc[2]*x*y + cc[3]*x + cc[4]*y + cc[5]))/Const[0][0]
    
    # vals2 = []
    # con = []
    # for i in range(len(xy.T)):
    #     xx = np.expand_dims(xy.T[i],axis=1)
    #     vals2.append(np.copy(np.exp(-(xx-scaling.mu).T@scaling.cov@(xx-scaling.mu))))
    #     con.append(-c[3]*xx[0] - c[4]*xx[1]-c[5])
        
    # vals2 = np.asarray(vals2)
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(mesh[:,0], mesh[:,1], pdf)
    # ax.scatter(x, y,np.expand_dims(np.exp(zobs),1)/np.expand_dims(gauss,1), c='r', marker='.')
    # ax.scatter(x, y,np.exp(zobs), c='k', marker='.')
    # ax.scatter(mesh[:,0], mesh[:,1], gauss2)
    # plot(xy, zobs, pred_params)
    # plt.show()
    return scaling, pdf/np.expand_dims(vals,1), pred_params, Const
    

def main():
    a, b, c, d, e, f = 0.1,0.1,0,0,0,0
    true_params = [a, b, c, d, e, f]
    xy, zobs = generate_example_data(6, true_params)
    x, y = xy

    i = zobs.argmax()
    guess = [1, 1, 0, 0, 0, 0]
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
    
    # gauss = fun.Gaussian(scaling, xy.T)
    # mesh = UM.generateOrderedGridCenteredAtZero(-.3, .3, -.3, .3, 0.01, includeOrigin=True)
    # gauss2 = fun.Gaussian(scaling, mesh)
    cc=pred_params
    x,y = xy
    vals = -(cc[0]*x**2+ cc[1]*y**2 + 2*cc[2]*x*y + cc[3]*x + cc[4]*y + cc[5])

    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(x, y,np.exp(zobs)/gauss, c='r', marker='.')
    
    print (np.exp(zobs)/vals)
    # ax.scatter(mesh[:,0], mesh[:,1], gauss2)
    # plot(xy, zobs, pred_params)
    # plt.show()
    

def quad(xy, a, b, c, d, e, f):
    x, y = xy
    A= np.asarray([[a,c],[c,b]])
    B=np.asarray([[d, e]]).T
    # quad = f + B.T@xy + xy.T@A@xy
    quad = -(a*x**2+ b*y**2 + 2*c*x*y + d*x + e*y + f)
    # penalty =0
    # penalty2 = 0
        
    # A= np.asarray([[a, c],[c,b]])
    # try:
    #     sigma = np.linalg.inv(A)
    # except:
    #     quad = quad -1000
    
    # sigma = np.linalg.inv(A)
    # if a*b-c**2 <= 0 or a<0 or b<0:
    #     quad = quad - abs(a*b-c**2)*100000

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
    scaling.setMu(np.asarray([[1,-1]]).T)
    scaling.setSigma(np.asarray([IC,IC]))
    zobs = np.log((UM.generateICPDF(xy.T[:,0], xy.T[:,1], IC,IC))**2)
    
    zobs = np.log(np.ones(len(xy.T)))
    
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
