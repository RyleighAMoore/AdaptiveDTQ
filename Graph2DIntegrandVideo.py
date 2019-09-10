# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 12:53:52 2019

@author: Ryleigh
"""

# Need to run 2DTQFast.py first.

import numpy as np
import random
import matplotlib.pyplot as plt


def random_color():
    rgbl=[255,0,0]
    random.shuffle(rgbl)
    return tuple(rgbl)


fig = plt.figure()
s = len(X)
ax = fig.add_subplot(111, projection='3d')
IntegrandOne = []
colors = iter(cm.rainbow(np.linspace(0, 1, s**2)))
for i in range(s**2):
    color =random_color()
    integrand = np.reshape(Integrands[5][:,i], [s, s])
    IntegrandOne.append(integrand)
    surf = ax.scatter(X, Y, integrand, color=next(colors))
fig.show()


def update_plot(frame_number, IntegrandOne, plot):
    plot[0].remove()
    plot[0] = ax.plot_surface(X, Y, IntegrandOne[frame_number], cmap="magma")


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
fps = 10  # frame per sec
frn = len(IntegrandOne)  # frame number of the animation

plot = [ax.plot_surface(X, Y, IntegrandOne[10], color='0.75', rstride=1, cstride=1)]
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlim(0, np.max(0.1))
ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(IntegrandOne, plot), interval=1000 / fps)
plt.title('fun.f1=x(4-r^2), fun.f2=y(4-r^2), fun.g1()=fun.g2()=1')
plt.show()
print(np.max(IntegrandOne[-1]))
t = 0
print(np.max(phat))