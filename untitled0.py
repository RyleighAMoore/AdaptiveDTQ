# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 23:19:17 2021

@author: Rylei
"""

pc = []
for i in range(len(Meshes)-1):
    l = len(Meshes[i+1])
    pc.append(LPReuseArr[i]/l)
    
mean = np.mean(pc[1:])

pc = []
for i in range(len(Meshes)-1):
    l = len(Meshes[i+1])
    pc.append(AltMethod[i]/l)
    
mean = np.mean(pc[1:])