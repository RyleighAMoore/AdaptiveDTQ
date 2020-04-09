
import numpy as np 

class GaussScale:
    def __init__(self, numVars):
        self.numVars = numVars
        self.mu = np.zeros((numVars))
        self.cov = np.zeros((numVars, numVars))
        self.sigma = np.sqrt(np.diagonal(self.cov))


