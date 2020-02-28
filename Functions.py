import numpy as np

# Drift fuction
def driftfun(x):
#     if isinstance(x, int) | isinstance(x, float):
#         return 1
#     else:
#         return np.ones(np.shape(x)) * 1
    return x * (4 - x ** 2)

# Diffusion ction
def difffun(x):
    return np.repeat(1, np.size(x))


# Density, distribution ction, quantile ction and random generation for the
# normal distribution with mean equal to mu and standard deviation equal to sigma.
def dnorm(x, mu, sigma):
    return np.divide(1, (sigma * np.sqrt(2 * np.pi))) * np.exp(np.divide(-(x - mu) ** 2, 2 * sigma ** 2))


def dnorm_partialx(x, mu, sigma):
    return np.divide(-x+mu,sigma**2)*np.divide(1, (sigma * np.sqrt(2 * np.pi))) * np.exp(np.divide(-(x - mu) ** 2, 2 * sigma ** 2))


# Volcano
# def f1(x, y):
#     r = np.sqrt(x ** 2 + y ** 2)
#     #return 1
#     return x * (3- r ** 2)


# def f2(x, y):
#     r = np.sqrt(x ** 2 + y ** 2)
#     #return 0
#     return y * (1- r ** 2) 

# def g1():
#     return 1

# def g2():
#     return 1
    
    
def f1(x, y):
    return 4

def f2(x, y):
    return 0

def g1():
    return 0.5

def g2():
    return 0.5

# def f1(x, y):
#     r = np.sqrt(x ** 2 + y ** 2)
#     #return 1
#     return x * (4- r ** 2)


# def f2(x, y):
#     r = np.sqrt(x ** 2 + y ** 2)
#     #return 1
#     return y * (1- r ** 2)

# def g1():
#     return 1

# def g2():
#     return 1

## Penguin
# def f1(x, y):
#     r = np.sqrt(x ** 2 + y ** 2)
#     return np.sin(x*r)


# def f2(x, y):
#     r = np.sqrt(x ** 2 + y ** 2)
#     return np.tan(y*r)**2

# def g1():
#     return 1

# def g2():
#     return 1
    
## Monster Truck
#def f1(x, y):
#    r = np.sqrt(x ** 2 + y ** 2)
#    return np.sin(x*r)
#
#
#def f2(x, y):
#    r = np.sqrt(x ** 2 + y ** 2)
#    return np.tan(y*r)
#
#def g1():
#    return 1
#
#def g2():
#    return 1
    

## YXXX
#def f1(x, y):
#    r = np.sqrt(x ** 2 + y ** 2)
#    return np.sin(x**2)
#
#
#def f2(x, y):
#    r = np.sqrt(x ** 2 + y ** 2)
#    return np.sin(x*r)
#
#def g1():
#    return 1
#
#def g2():
#    return 1
    
## Quad Hills
# def f1(x, y):
#     r = np.sqrt(x ** 2 + y ** 2)
#     return x*(1-x**4)

# def f2(x, y):
#     r = np.sqrt(x ** 2 + y ** 2)
#     return y*(1-y**4)

# def g1():
#     return 0.5

# def g2():
#     return 0.5
    
# def f1(x, y):
#     r = np.sqrt(x ** 2 + y ** 2)
#     return -x**2

# def f2(x, y):
#     r = np.sqrt(x ** 2 + y ** 2)
#     return x**2

# def g1():
#     return 0.5

# def g2():
#     return 0.5


def G(x1, x2, y1, y2, h):
    return ((2 * np.pi * g1() ** 2 * h) ** (-1 / 2) * np.exp(-(x1 - y1 - h * f1(y1, y2)) ** 2 / (2 * g1() ** 2 * h))) * (
                (2 * np.pi * g2() ** 2 * h) ** (-1 / 2) * np.exp(-(x2 - y2 - h * f2(y1, y2)) ** 2 / (2 * g2() ** 2 * h)))

def G1D(x1, x2, y1, gamma1, h):
    return (1 / (np.sqrt(2 * np.pi * gamma1**2 * h))) * np.exp((-(x1 - y1 - h * f1(y1, x2 + f2(x1,x2))) ** 2) / (2 * gamma1 ** 2 * h))

