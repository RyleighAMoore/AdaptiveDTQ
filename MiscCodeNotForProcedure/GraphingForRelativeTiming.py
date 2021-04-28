# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 10:46:21 2021

@author: Rylei
"""
import DTQAdaptive as D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import Functions as fun
from DriftDiffFunctionBank import MovingHillDrift, DiagDiffOne
from DTQTensorized import MatrixMultiplyDTQ
from exactSolutions import TwoDdiffusionEquation
from Errors import ErrorValsExact
from datetime import datetime

import pickle

L2wErrorArray  = pickle.load( open( "L2wErrorArray115.pickle", "rb" ) )
L2wErrorArrayT  = pickle.load( open( "L2wErrorArrayT115.pickle", "rb" ) )

TimingArray  = pickle.load( open( "TimingArray115.pickle", "rb" ) )
TimingArrayT  = pickle.load( open( "TimingArrayT115.pickle", "rb" ) ) 
    
mm = min(min(TimingArrayT), min(TimingArray))
mm = TimingArray[0]

from matplotlib import rcParams

# Font styling
rcParams['font.family'] = 'serif'
rcParams['font.weight'] = 'bold'
rcParams['font.size'] = '18'
fontprops = {'fontweight': 'bold'}

m = np.max(np.asarray(TimingArrayT)/mm) 
# nearest_multiple = int(5 * round(m/5))
plt.figure()
# plt.yticks(np.arange(0, nearest_multiple+10, 5))
plt.semilogx(L2wErrorArrayT[:,-1],np.asarray(TimingArrayT)/mm, 'o-',label="Tensorized")
plt.semilogx(L2wErrorArray[:,-1],np.asarray(TimingArray)/mm, 'o:r', label="Adaptive")
plt.semilogx(L2wErrorArray[0,-1],np.asarray(TimingArray[0])/mm, '*k', label="Unit Time", markersize=10)
plt.ylabel("Relative Time")
plt.xlabel(r"$L_{2w}$ Error")
plt.legend()
# plt.xticks([10**(-7), 10**(-6), 10**(-5),  10**(-4), 10**(-3),  10**(-2),  10**(-1), 10**(0)])
plt.xticks([10**(-8), 10**(-6),  10**(-4),  10**(-2), 10**(0)])

plt.show()
    