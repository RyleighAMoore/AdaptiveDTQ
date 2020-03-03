# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 01:10:33 2020

@author: Rylei
"""


mesh = UM.generateOrderedGridCenteredAtZero(-2.5,2.5, -2.5, 2.5, 0.1, includeOrigin=True)
pdf = UM.generateICPDF(mesh[:,0], mesh[:,1], 0.5, 0.5)

# for i in range(len(pdf)):
#     r = mesh[i,0]**2+mesh[i,1]**2
#     pdf[i] = r
    # pdf[i] = (np.sin((mesh[i,0]+ mesh[i,1])))**2
fig = plt.figure()
ax = Axes3D(fig)
index =-1
ax.scatter(mesh[:,0], mesh[:,1], pdf, c='r', marker='.')

pkl_file = open("C:/Users/Rylei/Documents/SimpleDTQ/PickledData/PDFTraj1.p", "wb" ) 
pkl_file2 = open("C:/Users/Rylei/Documents/SimpleDTQ/PickledData/Mesh1.p", "wb" ) 

surfaces = []
surfaces.append(pdf)
import pickle  
pickle.dump(surfaces, pkl_file)
pickle.dump(mesh, pkl_file2)
pkl_file.close()
pkl_file2.close()