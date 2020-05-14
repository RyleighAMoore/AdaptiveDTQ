import numpy as np
import matplotlib.pyplot as plt

def map_to_canonical_space(user_samples, scale_parameters):
    L = np.linalg.cholesky((scale_parameters.cov))
    Linv = np.linalg.inv(L)    
    shiftedMesh = user_samples - scale_parameters.mu.T*np.ones(np.shape(user_samples))
    
    canonical_samples = (Linv @ shiftedMesh.T).T
    
    # s = user_samples.shape
    # if user_samples.ndim == 1:
    #     user_samples = np.expand_dims(user_samples,1)
    # numVars = np.size(user_samples,1)
    # canonical_samples = user_samples.copy()
    # for ii in range(numVars):
    #     loc,scale = scale_parameters.mu[ii][0], scale_parameters.getSigma()[ii]
    #     canonical_samples[:,ii] = ((user_samples[:,ii] - loc)/(np.sqrt(2)*scale))
    # canonical_samples = np.reshape(canonical_samples,s)
    
    # plt.figure()
    # plt.scatter(canonical_samples[:,0], canonical_samples[:,1])
    # plt.scatter(user_samples[:,0], user_samples[:,1], c='red')

    return canonical_samples

def map_from_canonical_space(user_samples, scale_parameters):
    s = user_samples.shape
    if user_samples.ndim == 1:
        user_samples = np.expand_dims(user_samples,1)
    numVars = np.size(user_samples,1)
    canonical_samples = user_samples.copy()
    for ii in range(numVars):
        loc,scale = scale_parameters.mu[ii][0], scale_parameters.getSigma()[ii]
        canonical_samples[:,ii] = canonical_samples[:,ii]*np.sqrt(2)*scale+loc
    canonical_samples = np.reshape(canonical_samples,s)
    return canonical_samples


    

