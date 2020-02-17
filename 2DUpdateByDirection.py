from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import Functions as fun
import Integrand
import Operations2D
import XGrid
from mpl_toolkits.mplot3d import Axes3D



def f1(x, y):
    r = np.sqrt(x ** 2 + y ** 2)
    return 1
    return x * (4 - x ** 2)


def f2(x, y):
    r = np.sqrt(x ** 2 + y ** 2)
    return 0
    return y * (4 - y ** 2)

# def g(x, y):
#     return 2
g1 = 1
g2 = 0.5

def G(x1, x2, y1, y2, h):
    return ((2 * np.pi * g1 ** 2 * h) ** (-1 / 2) * np.exp(-(x1 - y1 - h * f1(y1, y2)) ** 2 / (2 * g1 ** 2 * h))) * (
                (2 * np.pi * g2 ** 2 * h) ** (-1 / 2) * np.exp(-(x2 - y2 - h * f2(y1, y2)) ** 2 / (2 * g2 ** 2 * h)))


def G1D(x1, x2, y1, gamma1):
    return (1 / (np.sqrt(2 * np.pi * gamma1**2 * h))) * np.exp((-(x1 - y1 - h * f1(y1, x2 + f2(x1,x2))) ** 2) / (2 * gamma1 ** 2 * h))


T = 0.01  # final time, code computes PDF of X_T
s = 0.75  # the exponent in the relation k = h^s
h = 0.01  # temporal step size
init = 0  # initial condition X_0
numsteps = int(np.ceil(T / h))

assert numsteps > 0, 'The variable numsteps must be greater than 0'

# define spatial grid
kstep = h ** s
kstep = 0.1
# kstep = 0.1
xMin1 = -2
xMax1 = 2
xMin2 = -2
xMax2 = 2

epsilonTol = -5


#x1 = np.arange(xMin1, xMax1, kstep)
#x2 = np.arange(xMin2, xMax2, kstep)
x1=XGrid.getCenteredZeroXvec(kstep,1)
x2=XGrid.getCenteredZeroXvec(kstep,1)

X, Y = np.meshgrid(x1, x2)

w1 = Operations2D.find_nearest(x1, 0)
w2 = Operations2D.find_nearest(x2, 0)


phat = np.zeros([len(x1), len(x2)])
a1 = init + f1(init,0)
b1 = np.abs(g1 * np.sqrt(h))
a2 = init + f2(init,0)
b2 = np.abs(g2 * np.sqrt(h))
phat0 = fun.dnorm(x1, a1, b1)  # pdf after one time step with Dirac \delta(x-init)
phat1 = fun.dnorm(x2, a2, b2)  # pdf after one time step with Dirac \delta(x-init)

#phat[w1, :] = phat1
#phat[:, w2] = phat0
phat[w1,w2]= 1/kstep**2



surfaces = []
Xs= []
Ys=[]
surfaces.append(np.matrix.transpose(np.copy(phat)))
Xs.append(np.copy(X))
Ys.append(np.copy(Y))
inds = np.asarray(list(range(0, np.size(x1)*np.size(x2))))
phat_rav = np.ravel(phat)

inds_unrav = np.unravel_index(inds, (len(x1), len(x2)))

val = []
val2=[]
        
if g2 == 0:
    Gmat = np.zeros([len(x1), len(x2)])
    print(0)
    for i in range(0, len(x1)):
        print(i)
        print(len(inds_unrav[0]))            
        for k in range(0, len(x1)):
            Gmat[i,k]=kstep*G1D(x1[i], x2[i], x1[k], g1)
            
    t=0
    while t < 150:
        print(t)
        phatMat = np.matmul(Gmat, phat)
        phat = phatMat
        t = t+1
        surfaces.append(np.matrix.transpose(np.copy(phatMat)))
            
if (g1 != 0) & (g2 != 0):
    Gmat = np.zeros([len(inds_unrav[0]), len(inds_unrav[1])])
    for i in range(0, len(inds_unrav[0])): # I
        print(i)
        print(len(inds_unrav[0]))            
        for k in range(0, len(inds_unrav[0])): # K
            Gmat[i,k]=kstep**2*G(x1[inds_unrav[0][i]], x2[inds_unrav[1][i]], x1[inds_unrav[0][k]], x2[inds_unrav[1][k]], h)

    t=0
    while t < 20:
        print(t)
        epsilon, vals = Operations2D.computeEpsilon(Gmat, phat_rav)
        print(epsilon)
        
        while epsilon > epsilonTol: # Need to extend the phat  
            print('again')
            print(vals)
            if vals[0] > epsilonTol: #First row
                loc, x1 = XGrid.addValueToXvec(x1, x1[0] - kstep)
                phatMat = np.vstack((np.zeros([1,len(x2)]),phatMat))
                
            if vals[1] > epsilonTol: # Last Row
                loc, x1 = XGrid.addValueToXvec(x1, x1[-1] + kstep)
                phatMat = np.vstack((phatMat, np.zeros([1,len(x2)])))

            if vals[2] > epsilonTol: # First Col
                loc, x2 = XGrid.addValueToXvec(x2, x2[0] - kstep)
                phatMat = np.hstack((np.zeros([len(x1),1]),phatMat))
                
            if vals[3] > epsilonTol: # Last Col
                loc, x2 = XGrid.addValueToXvec(x2, x2[-1] + kstep)
                phatMat = np.hstack((phatMat,np.zeros([len(x1),1])))

            inds = np.asarray(list(range(0, np.size(x1)*np.size(x2))))
            phat_rav = np.ravel(phatMat)
            inds_unrav = np.unravel_index(inds, (len(x1), len(x2)))
    
            Gmat = np.zeros([len(inds_unrav[0]),len(inds_unrav[0])])
            for i in range(0, len(inds_unrav[0])): # I
                for k in range(0, len(inds_unrav[0])): # K
                    Gmat[i,k]=kstep**2*G(x1[inds_unrav[0][i]], x2[inds_unrav[1][i]], x1[inds_unrav[0][k]], x2[inds_unrav[1][k]], h)
            
            epsilon, vals = Operations2D.computeEpsilon(Gmat, phat_rav)

        phat_rav = np.matmul(Gmat, phat_rav)
        phatMat = np.reshape(phat_rav,(len(x1),len(x2)))
        
        X, Y = np.meshgrid(x1, x2)
        Xs.append(np.copy(X))
        Ys.append(np.copy(Y))
        surfaces.append(np.matrix.transpose(np.copy(phatMat)))
        t = t+1



def update_plot(frame_number, surfaces, plot):
    plot[0].remove()
    plot[0] = ax.plot_surface(Xs[frame_number], Ys[frame_number], surfaces[frame_number], cmap="magma")


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
fps = 10  # frame per sec
frn = len(surfaces)  # frame number of the animation

plot = [ax.plot_surface(Xs[10], Ys[10], surfaces[10], color='0.75', rstride=1, cstride=1)]
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlim(0, np.max(np.max(surfaces[10])))
ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(surfaces, plot), interval=1000 / fps)
plt.title('f1=1, f2=0, g1= 1, g2=0.5')
plt.show()
print(np.max(surfaces[-1]))
t = 0
print(np.max(phat))
# t = 0
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(X, Y, pdf, rstride=1, cstride=1, antialiased=True)
# plt.show()