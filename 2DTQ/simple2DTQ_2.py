import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib
from pylab import *
import numpy as np
import numpy as np
print('numpy: ' + np.version.full_version)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def f1(x, y):
    r = np.sqrt(x ** 2 + y ** 2)
    return x*(0.4-r**2)
    return y*(1-y**2)

def f2(x, y):
    r = np.sqrt(x ** 2 + y ** 2)
    #return 0
    return x*(0.4-r**2)

# def g(x, y):
#     return 2

def G(x1, x2, y1, y2, h):
    g1 = 1
    g2 = 1
    return ((2*np.pi*g1**2*h)**(-1/2)*np.exp(-(x1-y1-h*f1(y1,y2))**2/(2*g1**2*h)))*((2*np.pi*g2**2*h)**(-1/2)*np.exp(-(x2-y2-h*f2(y1,y2))**2/(2*g2**2*h)))


T = 1  # final time, code computes PDF of X_T
s = 0.75  # the exponent in the relation k = h^s
h = 0.01  # temporal step size
init = 0  # initial condition X_0
numsteps = int(np.ceil(T / h))

assert numsteps > 0, 'The variable numsteps must be greater than 0'

# define spatial grid
kstep = h ** s
kstep = 0.1
xMin = -1.2
xMax = 1.2

x1 = np.arange(xMin, xMax, kstep)
x2 = np.arange(xMin, xMax, kstep)
# x1 = np.linspace(xMin, xMax, 23)
# x2 = np.linspace(xMin, xMax, 23)

X, Y = np.meshgrid(x1, x2)
phat = np.zeros([len(x1), len(x2)])
pdf = np.zeros([len(x1), len(x2)])

le = int(np.ceil(len(x1) / 2))-1
w = find_nearest(x1,0)
phat[w, w] = 1*(1/kstep**2)

t = 0
surfaces = []
surfaces.append(phat)
w=[]
while t < 10:
    print(t)
    for (ind_i, i) in enumerate(x1):
        #print(i)
        for (ind_j, j) in enumerate(x2):
            result2 = 0
            for (ind_k, k) in enumerate(x1):
                result = 0
                for (ind_l, l) in enumerate(x2):
                    ww = G(i, j, k, l, h)
                    w.append(ww)
                    result += kstep*G(i, j, k, l, h)*phat[ind_k,ind_l]
                result2 += kstep*result
            pdf[ind_i, ind_j] = result2
    phat = pdf
    surfaces.append(copy(pdf))
    t += 1

t=0

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
ax.set_zlim(0, np.max(np.max(surfaces[5])))
ani = animation.FuncAnimation(fig, update_plot,frn , fargs=(surfaces, plot), interval=1000 / fps)

plt.show()
print(np.max(surfaces[-1]))
t=0
# t = 0
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(X, Y, pdf, rstride=1, cstride=1, antialiased=True)
# plt.show()
