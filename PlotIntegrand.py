# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 18:28:12 2020

@author: Rylei
"""
import numpy as np
import Functions as fun
import UnorderedMesh as UM
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
from Functions import g1, g2, f1, f2

def newIntegrand(x1,x2,mesh,h):
    y1 = mesh[:,0]
    y2 = mesh[:,1]
    scale = h*g1(x1,x2)*g2(x1,x2)/(h*g1(y1,y2)*g2(y1,y2))
    val = scale*np.exp(-(h**2*f1(y1,y2)**2+2*h*f1(y1,y2)*(x1-y1))/(2*h*g1(x1,x2)**2) + -(h**2*f2(y1,y2)**2+2*h*f2(y1,y2)*(x2-y2))/(2*h*g2(x1,x2)**2))*np.exp((x1-y1+h*f1(y1,y2))**2/(2*h*g1(x1,x2)**2) - (x1-y1+h *f1(y1,y2))**2/(2*h*g1(y1,y2)**2) + (x2-y2+h*f2(y1,y2))**2/(2*h*g2(x1,x2)**2) - (x2-y2+h* f2(y1,y2))**2/(2*h*g2(y1,y2)**2))
    val = scale*np.exp(-(h**2*f1(y1,y2)**2-2*h*f1(y1,y2)*(x1-y1))/(2*h*g1(x1,x2)**2) + -(h**2*f2(y1,y2)**2-2*h*f2(y1,y2)*(x2-y2))/(2*h*g2(x1,x2)**2))*np.exp((x1-y1-h*f1(y1,y2))**2/(2*h*g1(x1,x2)**2) - (x1-y1-h *f1(y1,y2))**2/(2*h*g1(y1,y2)**2) + (x2-y2-h*f2(y1,y2))**2/(2*h*g2(x1,x2)**2) - (x2-y2-h* f2(y1,y2))**2/(2*h*g2(y1,y2)**2))

    return val

Px =0
Py = 0
h=0.01

mesh = UM.generateOrderedGridCenteredAtZero(-0.2, 0.2, -0.2, 0.2, 0.01, includeOrigin=True)
vals = []
for i in range(len(mesh)):
    value = fun.G(Px, Py, mesh[i,0], mesh[i,1], h)
    vals.append(value)
    
muX = Px #+ h*fun.f1(Px,Py)
muY = Py #+ h*fun.f2(Px,Py)
sigmaX = np.sqrt(h)*fun.g1()
sigmaY = np.sqrt(h)*fun.g2()
rv = multivariate_normal([muX, muY], [[sigmaX**2, 0], [0, sigmaX**2]])
gauss = np.asarray([rv.pdf(mesh)]).T

# Lg, tt = u13.calculateLg(Px, Py, mesh, h)
# Lf,ttt = u13.calculateLf(Px, Py, mesh, h)
vals = newIntegrand(Px, Py, mesh, h)
    
fig = plt.figure()
ax = Axes3D(fig)
ax.plot(mesh[:,0], mesh[:,1], np.asarray(vals)*np.squeeze(gauss), '*r', label='new integrand x Gaussian')
# ax.plot(mesh[:,0], mesh[:,1], np.asarray(vals), '*r', label='new part of integrand')

# ax.scatter(mesh[:,0], mesh[:,1], np.squeeze(gauss), c='g', label='Gaussian')
# ax.legend()
# ax.scatter(Px, Px, 0, c='r')

