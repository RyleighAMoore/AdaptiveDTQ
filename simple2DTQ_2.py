from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import Functions as fun
from mpl_toolkits.mplot3d import Axes3D



def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def f1(x, y):
    r = np.sqrt(x ** 2 + y ** 2)
    return x * (1 - x ** 2)
#    return x * (4-x ** 2)


def f2(x, y):
    r = np.sqrt(x ** 2 + y ** 2)
    # return 0.1
    #return 0
    return x * (1 - x ** 2)


# def g(x, y):
#     return 2
g1 = 1
g2 = 1


def G(x1, x2, y1, y2, h):
    return ((2 * np.pi * g1 ** 2 * h) ** (-1 / 2) * np.exp(-(x1 - y1 - h * f1(y1, y2)) ** 2 / (2 * g1 ** 2 * h))) * (
                (2 * np.pi * g2 ** 2 * h) ** (-1 / 2) * np.exp(-(x2 - y2 - h * f2(y1, y2)) ** 2 / (2 * g2 ** 2 * h)))


def G1D(x1, x2, y1, gamma1):
    return (1 / (np.sqrt(2 * np.pi * gamma1**2 * h))) * np.exp((-(x1 - y1 - h * f1(y1, x2)) ** 2) / (2 * gamma1 ** 2 * h))


T = 0.01  # final time, code computes PDF of X_T
s = 0.75  # the exponent in the relation k = h^s
h = 0.01  # temporal step size
init = 0  # initial condition X_0
numsteps = int(np.ceil(T / h))

assert numsteps > 0, 'The variable numsteps must be greater than 0'

# define spatial grid
kstep = h ** s
kstep =0.11
xMin = -1
xMax = 1

#x1 = np.arange(xMin, xMax, kstep)
#x2 = np.arange(xMin, xMax, kstep)
#x1 = np.linspace(xMin, xMax, 23)
#x2 = np.linspace(xMin, xMax, 23)
x1 = [0]
x2 = [0]
for step in range(1, 15):
     x1.append(0 + kstep * step)
     x1.append(0 - kstep * step)
x1 = np.sort(np.asarray(x1))
x2 = x1

X, Y = np.meshgrid(x1, x2)
phat = np.zeros([len(x1), len(x2)])
pdf = np.zeros([len(x1), len(x2)])
print(size(pdf, 1))

#le = int(np.ceil(len(x1) / 2)) - 1
w = find_nearest(x1, 0)
#ww = x1[w]
# phat[w+1, w] = 1*(1/(3*kstep**2))
# phat[w-1, w] = 1*(1/(3*kstep**2))
#phat[w, w] = (1 / (kstep ** 2))
# phat[w+1, w+20] = (1 / (2 * kstep ** 2))
# phat[w-1, w-20] = (1 / (2 * kstep ** 2))

a = init + f1(init,0)
b = np.abs(g1 * np.sqrt(h))
phat0 = fun.dnorm(x1, a, b)  # pdf after one time step with Dirac \delta(x-init)
phat[:, w] = phat0

#plt.figure()
#plt.plot(x1, phat[:, w] ,'.')
#plt.show()
#phat[:,w] = np.exp(-10*x1**2)

t = 0
surfaces = []
qvals  = np.zeros([len(x1), len(x2)])



surfaces.append(np.matrix.transpose(np.copy(phat)))
if g2 == 0:
    ee = 0
    while ee < 5:
        print(np.max(phat))
        print(ee)
        for (ind_i, i) in enumerate(x1):
            for (ind_j, j) in enumerate(x2):
                result = 0
                for (ind_k, k) in enumerate(x1):
                    q = G1D(i, j, k, g1) * phat[ind_k, ind_j]
                    result = result + q
                pdf[ind_i, ind_j] = kstep*result
        phat = np.copy(pdf)
        surfaces.append(np.matrix.transpose(copy(pdf)))
        ee += 1

else:
    while t < 5:
        print(t)
        for (ind_i, i) in enumerate(x1):
            #print(i)
            for (ind_j, j) in enumerate(x2):
                result2 = 0
                for (ind_k, k) in enumerate(x1):
                    result = 0
                    for (ind_l, l) in enumerate(x2):
                        result += kstep * G(i, j, k, l, h) * phat[ind_k, ind_l]
                    result2 += kstep * result
                pdf[ind_i, ind_j] = result2
        phat = np.copy(pdf)
        surfaces.append(np.matrix.transpose(copy(pdf)))
        t += 1
    t = 0


def update_plot(frame_number, surfaces, plot):
    plot[0].remove()
    plot[0] = ax.plot_surface(X, Y, surfaces[frame_number], cmap="magma")


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
fps = 10  # frame per sec
frn = len(surfaces)  # frame number of the animation

plot = [ax.plot_surface(X, Y, surfaces[1], color='0.75', rstride=1, cstride=1)]
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlim(0, np.max(np.max(surfaces[3])))
ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(surfaces, plot), interval=1000 / fps)

plt.show()
print(np.max(surfaces[-1]))
t = 0
print(np.max(phat))
# t = 0
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(X, Y, pdf, rstride=1, cstride=1, antialiased=True)
# plt.show()
