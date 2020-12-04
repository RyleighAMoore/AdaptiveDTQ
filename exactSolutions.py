import numpy as np

#Solution to Dd^2p/dx^2 + Dd^2p/dy^2 - dp/dt=0
def TwoDdiffusionEquation(mesh, D, t, A):
    D = D**2*0.5
    r = (mesh[:,0]-A*t)**2 +(mesh[:,1])**2
    rshift = np.sqrt((mesh[:,0]-t)**2 + (mesh[:,1]-t)**2)
    vals = np.exp(-r/(4*D*t))*(1/(4*np.pi*D*t))
    return vals

    
# ana = TwoDdiffusionEquation(mesh, 1,0.01)
# ana2 = TwoDdiffusionEquation(mesh, 1,0.1)
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(mesh[:,0], mesh[:,1], ana, c='k', marker='.')
# ax.scatter(mesh[:,0], mesh[:,1], ana2, c='r', marker='.')

