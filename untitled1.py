# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 17:48:20 2020

@author: Rylei
"""
import matplotlib.pyplot as plt

plt.figure()
plt.plot(SlopesMean)
plt.plot(SlopesMin)
plt.plot(SlopesMax, '.r')
plt.show()


plt.figure()
for i in range(len(Slopes)):
    one = i*np.ones(len(Slopes[i]))
    plt.scatter(one,Slopes[i], s=1)
plt.show()  

index=-2
fig = plt.figure()
ax = Axes3D(fig)
index = 0
ax.scatter(Meshes[31][:,0], Meshes[31][:,1], Slopes[31], c='r', marker='.')