import numpy as np

# Drift function
def driftfun(x):
    if isinstance(x, int) | isinstance(x, float):
        return 1
    else:
        return np.ones(np.shape(x)) * 1
    return x * (4 - x ** 2)


# Diffusion function
def difffun(x):
    return np.repeat(0, np.size(x))


# Density, distribution function, quantile function and random generation for the
# normal distribution with mean equal to mu and standard deviation equal to sigma.
def dnorm(x, mu, sigma):
    return np.divide(1, (sigma * np.sqrt(2 * np.pi))) * np.exp(np.divide(-(x - mu) ** 2, 2 * sigma ** 2))


def dnorm_partialx(x, mu, sigma):
    return np.divide(-x+mu,sigma**2)*np.divide(1, (sigma * np.sqrt(2 * np.pi))) * np.exp(np.divide(-(x - mu) ** 2, 2 * sigma ** 2))
