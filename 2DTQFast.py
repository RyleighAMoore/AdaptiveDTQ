from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import Functions as fun
import Integrand
import Operations2D
import XGrid
from mpl_toolkits.mplot3d import Axes3D

T = 0.01  # final time, code computes PDF of X_T
s = 0.75  # the exponent in the relation k = h^s
h = 0.01  # temporal step size
init = 0  # initial condition X_0
numsteps = int(np.ceil(T / h))

assert numsteps > 0, 'The variable numsteps must be greater than 0'

# define spatial grid
kstep = h ** s
kstep = 0.15
# kstep = 0.1
xMin1 = -3
xMax1 = 3
xMin2 = -3.2
xMax2 = 3.2

epsilonTol = -20


x1 = np.arange(xMin1, xMax1, kstep)
x2 = np.arange(xMin2, xMax2, kstep)

X, Y = np.meshgrid(x1, x2)

w1 = Operations2D.find_nearest(x1, 0)
w2 = Operations2D.find_nearest(x2, 0)


phat = np.zeros([len(x1), len(x2)])
a1 = init + fun.f1(init,0)
b1 = np.abs(fun.g1() * np.sqrt(h))
a2 = init + fun.f2(init,0)
b2 = np.abs(fun.g2() * np.sqrt(h))
phat0 = fun.dnorm(x1, a1, b1)  # pdf after one time step with Dirac \delta(x-init)
phat1 = fun.dnorm(x2, a2, b2)  # pdf after one time step with Dirac \delta(x-init)

phat[w1, :] = phat1
phat[:, w2] = phat0



surfaces = []
surfaces.append(np.matrix.transpose(np.copy(phat)))

inds = np.asarray(list(range(0, np.size(x1)*np.size(x2))))
phat_rav = np.ravel(phat)

inds_unrav = np.unravel_index(inds, (len(x1), len(x2)))

val = []
val2=[]

        
if fun.g2() == 0:
    Gmat = np.zeros([len(x1), len(x2)])
    print(0)
    for i in range(0, len(x1)):
        print(i)
        print(len(inds_unrav[0]))            
        for k in range(0, len(x1)):
            Gmat[i,k]=kstep*fun.G1D(x1[i], x2[i], x1[k], fun.g1(),h)
            
    t=0
    while t < 150:
        print(t)
        phatMat = np.matmul(Gmat, phat)
        phat = phatMat
        t = t+1
        surfaces.append(np.matrix.transpose(np.copy(phatMat)))
            
if (fun.g1() != 0) & (fun.g2() != 0):
    Gmat = np.zeros([len(inds_unrav[0]), len(inds_unrav[1])])
    for i in range(0, len(inds_unrav[0])): # I
        print(i)
        print(len(inds_unrav[0]))            
        for k in range(0, len(inds_unrav[0])): # K
            Gmat[i,k]=kstep**2*fun.G(x1[inds_unrav[0][i]], x2[inds_unrav[1][i]], x1[inds_unrav[0][k]], x2[inds_unrav[1][k]], h)

    t=0
    while t < 120:
        print(t)
        phat_rav = np.matmul(Gmat, phat_rav)
        phatMat = np.reshape(phat_rav,(len(x1),len(x2))) 
        t = t+1
        surfaces.append(np.matrix.transpose(np.copy(phatMat)))
    


def update_plot(frame_number, surfaces, plot):
    plot[0].remove()
    plot[0] = ax.plot_surface(X, Y, surfaces[frame_number], cmap="magma")


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
fps = 10  # frame per sec
frn = len(surfaces)  # frame number of the animation

plot = [ax.plot_surface(X, Y, surfaces[5], color='0.75', rstride=1, cstride=1)]
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlim(0, np.max(np.max(surfaces[5])))
ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(surfaces, plot), interval=1000 / fps)
plt.title('fun.f1=x(4-r^2), fun.f2=y(4-r^2), fun.g1()=fun.g2()=1')
plt.show()
print(np.max(surfaces[-1]))
t = 0
print(np.max(phat))
# t = 0
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(X, Y, pdf, rstride=1, cstride=1, antialiased=True)
# plt.show()
