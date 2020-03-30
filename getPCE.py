from pyapprox.variables import IndependentMultivariateRandomVariable
from scipy.stats import norm, uniform
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

indices20 = get_total_degree_indices(20, 2)
indices30 = get_total_degree_indices(30, 2)
indices50 = get_total_degree_indices(50, 2)
def generatePCE(degree, muX=0, muY=0, sigmaX=1, sigmaY=1):
    univariate_variables = [norm(muX,sigmaX),norm(muY,sigmaY)]
    variable = IndependentMultivariateRandomVariable(univariate_variables)
    var_trans = AffineRandomVariableTransformation(variable)
    
    poly = PolynomialChaosExpansion()
    poly_opts = define_poly_options_from_variable_transformation(var_trans)
    poly.configure(poly_opts)
    if degree ==20:
        indices = indices20
    elif degree == 30: 
        indices = indices30
    elif degree == 50:
        indices = indices50
    else:
        print("No indices matches the degree")
      # indices = compute_hyperbolic_indices(poly.num_vars(),degree,1.0)
    
        indices = get_total_degree_indices(degree, poly.num_vars())
    
    # indices = compute_tensor_product_level_indices(poly.num_vars(),degree,max_norm=True)
    poly.set_indices(indices)
    return poly


def generatePCE_Uniform(degree):
    univariate_variables = [uniform(),uniform()]
    variable = IndependentMultivariateRandomVariable(univariate_variables)
    var_trans = AffineRandomVariableTransformation(variable)
    
    poly = PolynomialChaosExpansion()
    poly_opts = define_poly_options_from_variable_transformation(var_trans)
    poly.configure(poly_opts)
    
    indices = compute_hyperbolic_indices(poly.num_vars(),degree,1.0)
    
    indices = get_total_degree_indices(degree, poly.num_vars())
    # indices = compute_tensor_product_level_indices(poly.num_vars(),degree,max_norm=True)
    poly.set_indices(indices)
    return poly


# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from scipy.integrate import simps
# import chaospy

# # expansion1 = chaospy.orth_ttr(2, chaospy.Normal(1, .1))
# distribution = chaospy.J(chaospy.Normal(0, 1), chaospy.Normal(0, 1))
# poly2 = chaospy.orth_ttr(4, distribution)

# nx = 20
# x = np.linspace(-6, 6, nx)
# xs= np.vstack((x,x))

# from scipy.stats import multivariate_normal
# var = 1
# poly = generatePCE(3,sigmaX=var, sigmaY=var)
# rv = multivariate_normal([0, 0], [[var, 0], [0, var]])
# weights = []
# vals = []
# vals2 = []
# xs = []
# ys = []
# for i in x:
#     for k in x:
#         point = np.vstack((i,k))
#         LP = poly.basis_matrix(point)
#         lp = LP[:,8]
#         lp2 = poly2(i,k)[8]
#         weights.append(np.asarray([rv.pdf(point.T)]).T)
#         vals.append(np.copy(lp))
#         vals2.append(lp2)
#         xs.append(i)
#         ys.append(k)
 
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(np.asarray(xs), np.asarray(ys), np.asarray(vals), c='k', marker='o')
# ax.scatter(np.asarray(xs), np.asarray(ys), np.asarray(vals2), c='r', marker='.')

        
        
# # poly1 = np.reshape(np.asarray(vals),(len(x),len(x)))
# # weights = np.reshape(np.asarray(weights),(len(x),len(x)))

# # fig = plt.figure()
# # ax = Axes3D(fig)
# # ax.scatter(np.asarray(xs), np.asarray(ys), np.asarray(vals), c='k', marker='.')


# # poly = generatePCE(5,sigmaX=1, sigmaY=1)
# # point = np.vstack((0,0))
# # # poly = chaospy.orth_ttr(2, distribution)

# # vals = []
# # xs = []
# # ys = []
# # for i in x:
# #     for k in x:
# #         point = np.vstack((i,k))
# #         # LP = poly.basis_matrix(point)
# #         # lp = np.sum(LP, axis=1)[0]
# #         # lp = np.exp(k
# #         lp = np.sum(poly1(i,k))
# #         vals.append(np.copy(lp))
# #         xs.append(i)
# #         ys.append(k)
# # poly2 = np.reshape(np.asarray(vals),(len(x),len(x)))

# # fig = plt.figure()
# # ax = Axes3D(fig)
# # ax.scatter(np.asarray(xs), np.asarray(ys), np.asarray(vals), c='k', marker='.')

# # # We first integrate over x and then over y
# # print(simps([simps(zz_x,x) for zz_x in poly1*poly2*weights],x))
