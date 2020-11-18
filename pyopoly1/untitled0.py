# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 11:43:45 2020

@author: Rylei
"""
from sklearn.neighbors import KDTree
import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
kdt = KDTree(X, leaf_size=30, metric='euclidean')
kdt.query(mesh, k=30, return_distance=False)