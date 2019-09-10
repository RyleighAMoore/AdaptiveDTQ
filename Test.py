# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 11:23:44 2019

@author: Ryleigh
"""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import UnorderedMesh as UM
size = 100
sigma_x = 0.1
sigma_y = 0.1

x = np.linspace(-10, 10, size)
y = np.linspace(-10, 10, size)
mesh = UM.generateRandomPoints(-1,1,-1,1,10000)
x = mesh[:,0]
y = mesh[:,1]

def generateICPDF(x,y,sigma_x, sigma_y):
    z = (1/(2*np.pi*sigma_x*sigma_y) * np.exp(-(x**2/(2*sigma_x**2)
         + y**2/(2*sigma_y**2))))

#    fig = plt.figure()
#    ax = Axes3D(fig)
#    ax.scatter(x, y, z, c='r', marker='.')
    
    return z
