# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 14:04:20 2019

@author: Ryleigh
"""

diffs = []
for i in range(100):
    diff = max(pdf_trajectory[i]- surfaces[i][126,:])
    diffs.append(diff)
    
plt.figure()
plt.plot(diffs)
plt.show()