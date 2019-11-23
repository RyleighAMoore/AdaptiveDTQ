# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 10:38:41 2019

@author: Ryleigh
"""
import numpy as np

def checkIntegrandForZeroPoints(GMat, PDF, tolerance):
    maxMat = []
    for i in range(len(PDF)):
        maxMat.append(np.max(PDF[i]*GMat[i]))    
    possibleZeros = [np.asarray(maxMat) <= tolerance]
    return np.asarray(possibleZeros).T

# Removes the flagged values from the list of mesh values and in Gmat. 
# boolZerosArray is the list of zeros and ones denoting which grid points to remove.
# Gmat, Mesh, Grids, Vertices, and VerticesNum are all used in the 2DTQ-UnorderedMesh method 
# and the parts associated with the removed points need to be removed.
def removePointsFromMesh(boolZerosArray,GMat, Mesh, Grids, Vertices, VerticesNum, Pdf):
    ChangedBool = 0
    if max(boolZerosArray == 1):
        for val in range(len(boolZerosArray)-1,-1,-1):
            print(val)
            if boolZerosArray[val] == 1: # remove the point
                GMat.pop(val)
                Mesh= np.delete(Mesh, val, 0)
                Grids.pop(val)
                Vertices.pop(val)
                VerticesNum.pop(val)
                Pdf = np.delete(Pdf, val, 0)
                print('removed point')
                ChangedBool=1
            
    return GMat, Mesh, Grids, Vertices, VerticesNum, Pdf, ChangedBool
            
    
        
        
        
    
    