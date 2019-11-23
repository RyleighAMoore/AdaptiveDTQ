# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 16:22:16 2019

@author: Ryleigh
"""

import numpy as np
import XGrid
import random
import matplotlib.pyplot as plt

for i in range(len(mesh)):
    for j in range(len(Grids[i])):
        plt.plot(Grids[j,0], Grids(j,1))
        plt.plot(Vertices[i][j][0])