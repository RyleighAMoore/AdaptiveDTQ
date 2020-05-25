# -*- coding: utf-8 -*-
"""
Created on Thu May 21 13:03:32 2020

@author: Rylei
"""
import pickle

pkl_file= open("C:/Users/Rylei/Documents/SimpleDTQ/PickledData/ICMesh1.p", "wb" ) 
pickle.dump(mesh, pkl_file)
pkl_file.close()