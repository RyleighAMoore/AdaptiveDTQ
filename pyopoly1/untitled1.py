# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 21:49:18 2020

@author: Rylei
"""
import math
import numpy as np

a,w = np.polynomial.hermite.hermgauss(32)

Ipeak = 4
sigmax = 0.2e-3
sigmay = 0.3e-3
sqrt2 = math.sqrt(2.)

def h(x, y):
    return Ipeak*1.0/math.pi

s = 0.0
for k in range(0, len(a)):
    x = sqrt2 * sigmax * a[k]
    t = 0.0
    for l in range(0, len(a)):
        y = sqrt2 * sigmay * a[l]
        t += w[l] * h(x, y)
    s += w[k]*t

print(s)