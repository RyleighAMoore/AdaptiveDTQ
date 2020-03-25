from pyapprox.variables import IndependentMultivariateRandomVariable
from scipy.stats import norm
from pyapprox.multivariate_polynomials import PolynomialChaosExpansion, define_poly_options_from_variable_transformation
from pyapprox.variable_transformations import AffineRandomVariableTransformation
from pyapprox.indexing import compute_hyperbolic_indices, argsort_indices_leixographically
import numpy as np
def get_total_degree_indices(degree, num_vars): 
    values = compute_hyperbolic_indices(num_vars,degree,1.0)
    ordering = argsort_indices_leixographically(values)
    xs = []
    ys = []
    for i in range(len(ordering)):
        xs.append(values[0,ordering[i]])
        ys.append(values[1,ordering[i]])
    z=np.vstack((xs,ys))
    return z 

def generatePCE(degree, muX=0, muY=0, sigmaX=1, sigmaY=1):
    univariate_variables = [norm(muX,sigmaX),norm(muY,sigmaY)]
    variable = IndependentMultivariateRandomVariable(univariate_variables)
    var_trans = AffineRandomVariableTransformation(variable)
    
    poly = PolynomialChaosExpansion()
    poly_opts = define_poly_options_from_variable_transformation(var_trans)
    poly.configure(poly_opts)
    
    indices = compute_hyperbolic_indices(poly.num_vars(),degree,1.0)
    
    # indices = get_total_degree_indices(degree, poly.num_vars())
    # indices = compute_tensor_product_level_indices(poly.num_vars(),degree,max_norm=True)
    poly.set_indices(indices)
    return poly



# poly,indices = generatePCE(20)

# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import numpy as np
# from matplotlib.animation import FuncAnimation,FFMpegFileWriter

# def update_graph(num):
#     xs = indices[:num,0]
#     ys = indices[:num,1]
#     graph = plt.set_data(indices[:num,0], indices[:num,1])
#     return graph


# fig = plt.figure()
# indices = indices.T  
# graph = plt.scatter(indices[:100,0], indices[:100,1])
# ani = animation.FuncAnimation(fig, update_graph, frames=indices,
#                                           interval=500, blit=False)

# plt.show()
