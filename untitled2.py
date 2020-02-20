# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:38:46 2020

@author: Rylei
"""

index=0
plt.figure()
plt.plot(Grids[index][:,0], Grids[index][:,1], '*')
plt.plot(Meshes[-1][index,0], Meshes[-1][index,1], '*r')
plt.plot(Meshes[-1][:,0], Meshes[-1][:,1], '.')

plt.show()