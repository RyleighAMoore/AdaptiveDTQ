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
    return x * (4 - r ** 2)


def f2(x, y):
    r = np.sqrt(x ** 2 + y ** 2)
    return y * (4 - r ** 2)

# def g(x, y):
#     return 2
g1 = 1
g2 = 1


def G(x1, x2, y1, y2, h):
    return ((2 * np.pi * g1 ** 2 * h) ** (-1 / 2) * np.exp(-(x1 - y1 - h * f1(y1, y2)) ** 2 / (2 * g1 ** 2 * h))) * (
                (2 * np.pi * g2 ** 2 * h) ** (-1 / 2) * np.exp(-(x2 - y2 - h * f2(y1, y2)) ** 2 / (2 * g2 ** 2 * h)))


def G1D(x1, x2, y1, gamma1):
    return (1 / (np.sqrt(2 * np.pi * gamma1**2 * h))) * np.exp((-(x1 - y1 - h * f1(y1, x2 + f2(x1,x2))) ** 2) / (2 * gamma1 ** 2 * h))

def G1(x1, x2, y1, y2, h):
    return ((2 * np.pi * g1 ** 2 * h) ** (-1 / 2) * np.exp(-(x1 - y1 - h * f1(y1, y2)) ** 2 / (2 * g1 ** 2 * h)))

def G2(x1, x2, y1, y2, h):
    return ((2 * np.pi * g2 ** 2 * h) ** (-1 / 2) * np.exp(-(x2 - y2 - h * f2(y1, y2)) ** 2 / (2 * g2 ** 2 * h)))

#def formB(x1,x2, h, blockNumber):
#    B = np.zeros([len(x1), len(x2)])
#    for i in range(len(x1)):
#        for j in range(len(x2)):
#            B[i,j] = G2(x1[blockNumber], x2[i], x1[blockNumber], x2[j], h)
#    return B
    
#def form_a(x1, x2, h, blockNumber):
#    a = np.zeros([len(x1), len(x2)])
#    for i in range(len(x1)):
#        for j in range(len(x2)):
#            a[i,j]=G1(x1[blockNumber], x2[i], x1[blockNumber], x2[j], h)
#    return a.T
    
def formB(x1, x2, h, blockNumber):
    B = np.zeros([len(x1), len(x2)])
    for i in range(len(x1)):
        for j in range(len(x2)):
            #B[i,j] = G2(x1[j], x2[i], x1[i], x2[j], h)
            B[i,j] = G2(x1[j], x2[i], x1[i], x2[j], h)
    return B


def form_a(x1, x2, h, blockNumber):
    a = np.zeros([len(x1), len(x2)])
    for i in range(len(x1)):
        for j in range(len(x2)):
            #a[i,j]=G1(x1[j], x2[i], x1[i], x2[j], h)
            a[i,j]=G1(x1[i], x2[j], x1[j], x2[i], h)
    return a

            
    
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
xMin = -3
xMax = 3

#x1 = [0]
#x2 = [0]
#for step in range(1, 30):
#     x1.append(0 + kstep * step)
#     x1.append(0 - kstep * step)
#x1 = np.sort(np.asarray(x1))
#x2 = x1

x1 = np.arange(xMin, xMax, kstep)
x2 = np.arange(xMin, xMax, kstep)

X, Y = np.meshgrid(x1, x2)

w = find_nearest(x1, 0)

phat = np.zeros([len(x1), len(x2)])
a = init + f1(init,0)
b = np.abs(g1 * np.sqrt(h))
phat0 = fun.dnorm(x1, a, b)  # pdf after one time step with Dirac \delta(x-init)
phat[:, w] = phat0

surfaces = []

surfaces.append(np.matrix.transpose(np.copy(phat)))

inds = np.asarray(list(range(0, np.size(phat))))
phat_rav = np.ravel(phat.T)

inds_unrav = np.unravel_index(inds, (len(phat), len(phat)))

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
    while t < 50:
        print(t)
        phatMat = np.matmul(Gmat, phat)
        phat = phatMat
        t = t+1
        surfaces.append(np.matrix.transpose(np.copy(phatMat)))
   
         
else:
    Gmat = np.zeros([len(inds_unrav[0]), len(inds_unrav[1])])
    runningSum = np.zeros([len(inds_unrav[0])])
#    for i in range(0, len(x1)):
#        for j in range(0,len(x2)):
#            a = G1(x1[i], x2[i], x1[i], x2[i], h)
#            B = formB(x1, x2, h, i)
#            G = a*B
    B = formB(x1, x2, h, 0)
    a = form_a(x1,x2, h, 0)
    Gmat = kstep**2*np.kron(a,B)
            

    t=0
    while t < 140:
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

plot = [ax.plot_surface(X, Y, surfaces[20], color='0.75', rstride=1, cstride=1)]
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlim(0, np.max(np.max(surfaces[20])))
ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(surfaces, plot), interval=1000 / fps)
plt.title('f1=x(4-r^2), f2=y(4-r^2), g1=g2=1')
plt.show()
print(np.max(surfaces[-1]))
t = 0
print(np.max(phat))
# t = 0
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(X, Y, pdf, rstride=1, cstride=1, antialiased=True)
# plt.show()