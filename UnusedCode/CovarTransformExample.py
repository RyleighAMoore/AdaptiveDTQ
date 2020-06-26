import sys
sys.path.insert(1, r'..')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import Functions as fun
from QuadratureUtils import *
import ICMeshGenerator as M

mean = [0, 0]
cov = [[9, 1], [1, 1]]  # diagonal covariance

import matplotlib.pyplot as plt
x, y = np.random.multivariate_normal(mean, cov, 5000).T

plt.plot(x, y, 'x')
plt.axis('equal')
plt.show()

         
L = np.linalg.cholesky((cov))

mesh = np.vstack((x,y)).T

trans = np.linalg.inv(L)@mesh.T
covNew = np.cov(trans)


SigmaInv = np.linalg.inv(cov)
vals1 = mesh@SigmaInv@mesh.T

x,y = trans
plt.scatter(x,y, color='red', marker='o')
x0,y0 = mesh.T
plt.scatter(x0,y0, color='black', marker='.')
plt.axis('equal')