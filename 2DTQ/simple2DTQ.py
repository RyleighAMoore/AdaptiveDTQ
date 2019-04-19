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


def f(x, y):
    r = np.sqrt(x ** 2 + y ** 2)
    return r*(4-r**2)
    #return 1


def g(x, y):
    return 1


def G(x1, x2, y1, y2, h):
    # 1 / (2 * np.pi * h * np.sqrt(g(y1) ** 2) * np.sqrt(g(y2) ** 2)) * np.exp(
    #   -0.5 * ((((x1 - (y1 + f(y1) * h)) ** 2) / h * g(y1) ** 2) + ((x2 - (y2 + f(y2) * h)) ** 2) / h * g(y2) ** 2))
    ys = np.sqrt(y1**2 + y2**2)
    ys = y1+y2
    xs = np.sqrt(x1**2 + x2**2)
    xs = x1+x2
    # return 1 / (2 * np.pi * h * np.sqrt(g(y1, y2) ** 2) * np.sqrt(g(y1, y2) ** 2)) * np.exp(
    #     -0.5 * ((((x1 - (y1-y2 + f(y1, y2) * h)) ** 2) / (h * g(y1, y2) ** 2)) + ((x2+x1 - (y2+y1 + f(y1, y2) * h)) ** 2) / (h * g(y1, y2) ** 2)))
    return 1 / (2 * np.pi * h * np.sqrt(g(y1, y2) ** 2) * np.sqrt(g(y1, y2) ** 2)) * np.exp(
        -0.5 * ((((xs - (ys + f(y1, y2) * h)) ** 2) / (h * g(y1, y2) ** 2)) + (
                    (xs - (ys + f(y1, y2) * h)) ** 2) / (h * g(
            y1, y2) ** 2)))



T = 1  # final time, code computes PDF of X_T
s = 0.75  # the exponent in the relation k = h^s
h = 0.01  # temporal step size
init = 0  # initial condition X_0
numsteps = int(np.ceil(T / h))

assert numsteps > 0, 'The variable numsteps must be greater than 0'

# define spatial grid
kstep = h ** s
kstep= 0.3
xMin = -2
xMax = 2

x1 = np.arange(xMin, xMax, kstep)
x2 = np.arange(xMin, xMax, kstep)
X, Y = np.meshgrid(x1, x2)
phat = np.zeros([len(x1), len(x2)])
pdf = np.zeros([len(x1), len(x2)])

le = int(np.ceil(len(x1) / 2) - 1)
phat[le, le] = 1
# phat[le+1, le]=1
# phat[le-1, le]=1
# phat[le, le+1]=1
# phat[le, le-1]=1

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(X, Y, phat, rstride=1, cstride=1, antialiased=True)
# plt.show()
t = 0
surfaces = []
surfaces.append(phat)
w=[]
while t < 2:
    for (ind_i, i) in enumerate(x1):
        print(i)
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
    surfaces.append(pdf)
    t += 1
    #minimum = np.min(w)

t=0
# for i in range(surfaces):
#     surfaces[:, :, i] = f(x, y, 1.5 + np.sin(i * 2 * np.pi / frn))

def update_plot(frame_number, surfaces, plot):
    plot[0].remove()
    plot[0] = ax.plot_surface(X, Y, surfaces[frame_number], cmap="magma")


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# fps = 1  # frame per sec
# frn = len(surfaces)  # frame number of the animation
#
# plot = [ax.plot_surface(X, Y, surfaces[1], color='0.75', rstride=1, cstride=1)]
# ax.set_zlim(0, np.max(np.max(surfaces[0])))
# ani = animation.FuncAnimation(fig, update_plot,frn , fargs=(surfaces, plot), interval=1000 / fps)
#
# plt.show()

t = 0
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, pdf, rstride=1, cstride=1, antialiased=True)
plt.show()
