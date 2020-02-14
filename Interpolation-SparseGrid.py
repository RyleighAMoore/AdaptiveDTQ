from pyapprox.sparse_grid import *
from pyapprox.visualization import plot_3d_indices
from pyapprox.univariate_quadrature import *
plt.rcParams['text.usetex'] = False
import unittest
from pyapprox.univariate_quadrature import *
from scipy.special import gamma as gamma_fn
from scipy.special import beta as beta_fn
from pyapprox.utilities import beta_pdf_on_ab, gaussian_pdf

from scipy.stats import multivariate_normal

num_vars = 2; level = 5
quad_rule = gaussian_leja_quadrature_rule
growth_rule = leja_growth_rule

samples, weights, data_structures = get_sparse_grid_samples_and_weights(
num_vars,level,quad_rule,growth_rule)

rv = multivariate_normal([0, 0], [[1, 0], [0, 1]])
pdf = rv.pdf(samples.T)
# y = multivariate_normal.pdf(samples.T, mean=0, cov=1)



plot_sparse_grid_2d(samples,weights)
plt.xlabel(r'$z_1$')
plt.ylabel(r'$z_2$')
plt.show()



from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(samples[0,:], samples[1,:], pdf, c='r', marker='.')

integral = np.dot(np.ones(len(weights)), weights)
    


