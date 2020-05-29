# code to compute the PDF of the solution of the SDE:
#
# dX_t = X_t*(4-X_t^2) dt + dW_t
#
# The deterministic ODE version is: dx/dt = x*(4-x^2)
# Examining this ODE, we see that there should be stable equilibria at x=+2 and x=-2.
# There is also an unstable equilibrium at x=0.

# drift and diffusion functions
driftfun <- function(x) { return(x*(4-x^2)) }
difffun <- function(x) { return(rep(1,length(x))) }

# function that returns kernel matrix
integrandmat <- function(xvec,yvec,h,driftfun,difffun)
{
  Y = t(replicate(length(xvec),yvec))
  mu = Y + driftfun(Y)*h
  sigma = abs(difffun(Y))*sqrt(h)
  test = dnorm(x=xvec, mean=mu, sd=sigma)
  return(test)
}

# simulation parameters
T = 1     # final time, code computes PDF of X_T
s = 0.75  # the exponent in the relation k = h^s
h = 0.01  # temporal step size
init = 0  # initial condition X_0
numsteps = ceiling(T/h)

# define spatial grid
k = h^s
yM = 0.05*k*(pi/(k^2))
xvec = seq(-yM,yM,by=k)

# kernel matrix
A = k*integrandmat(xvec,xvec,h,driftfun,difffun)

# pdf after one time step with Dirac \delta(x-init) initial condition
phat = as.matrix(dnorm(x=xvec, 
                       mean=(init+driftfun(init)), 
                       sd=abs(difffun(init))*sqrt(h)))

# main iteration loop
for (i in c(2:numsteps))
  phat = A %*% phat

# plot solution
# note that it is bimodal with modes at the stable equilibria, x=+2 and x=-2
plot(xvec, phat)
abline(v=2)
abline(v=-2)
