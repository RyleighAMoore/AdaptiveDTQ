import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Functions import *
import sys
sys.path.insert(1, r'C:\Users\Rylei\Documents\SimpleDTQ\pyopoly')
from Scaling import GaussScale
from variableTransformations import *
from Functions import covPart
import math


def fitGaussian(Px,Py, mesh, pdf):
    x0, y0 = Px, Py
    x, y = mesh.T
    xy = mesh.T
    zobs = np.squeeze(pdf)

    i = zobs.argmax()
    guess = [1, 0, 0, 75, 1, 75]
    pred_params, uncert_cov = opt.curve_fit(gauss2d, xy, zobs, p0=guess)

    zpred = gauss2d(xy, *pred_params)
    if math.isnan(zpred[3]) or math.isnan(zpred[5]):
        t=9

    scale = GaussScale(2)
    scale.setMu(np.asarray([[pred_params[1],pred_params[2]]]).T)
    print((1/(2*pred_params[3]))**(1/2), (1/(2*pred_params[5]))**(1/2))
    scale.setSigma([(1/(2*pred_params[3]))**(1/2), (1/(2*pred_params[5]))**(1/2)])
    A = pred_params[0]*(2*np.pi*(1/(2*pred_params[3]))**(1/2)* (1/(2*pred_params[5]))**(1/2))
    
    # pp = Gaussian(scale, mesh)*A
    cov = covPart(Px, Py, mesh, pred_params[4])
    
    
    return scale, A, zpred/cov, cov
    

def main():
    x0, y0 = 0, 0
    amp, a, b, c = 1, 0.1, 0, 0.1
    true_params = [amp, x0, y0, a, b, c]
    xy, zobs = generate_example_data(10, true_params)
    x, y = xy

    i = zobs.argmax()
    guess = [1, x[i], y[i], 1, 1, 1]
    pred_params, uncert_cov = opt.curve_fit(gauss2d, xy, zobs, p0=guess)

    zpred = gauss2d(xy, *pred_params)
    print('True parameters: ', true_params)
    print('Predicted params:', pred_params)
    print('Residual, RMS(obs - pred):', np.sqrt(np.mean((zobs - zpred)**2)))

    plot(xy, zobs, pred_params)
    plt.show()
    

def gauss2d(xy, amp, x0, y0, a, b, c):
    x, y = xy
    inner = a * (x - x0)**2
    inner += 2 * b * (x - x0)**2 * (y - y0)**2
    inner += c * (y - y0)**2
    print(np.max(-inner))
    return amp * np.exp(-inner)

def generate_example_data(num, params):
    np.random.seed(1977) # For consistency
    xy = np.random.random((2, num))

    noise = np.random.normal(0, 0.1, num) 
    zobs = gauss2d(xy, *params)
    return xy, zobs

def plot(xy, zobs, pred_params):
    x, y = xy
    yi, xi = np.mgrid[-1:1:30j, -1:1.2:30j]
    xyi = np.vstack([xi.ravel(), yi.ravel()])

    zpred = gauss2d(xyi, *pred_params)
    zpred.shape = xi.shape

    fig, ax = plt.subplots()
    ax.scatter(x, y, c=zobs, s=200, vmin=zpred.min(), vmax=zpred.max())
    ax.scatter(x, y, c='k', s=200)

    im = ax.imshow(zpred, extent=[xi.min(), xi.max(), yi.max(), yi.min()],
                   aspect='auto')
    fig.colorbar(im)
    ax.invert_yaxis()
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(xi, yi,zpred, c='r', marker='.')
    return fig

# main()
# 