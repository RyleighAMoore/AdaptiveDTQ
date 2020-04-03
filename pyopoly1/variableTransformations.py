# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 19:34:36 2020

@author: Rylei
"""
import numpy as np


def map_to_canonical_space(user_samples, scale_parameters):
    s = user_samples.shape
    if user_samples.ndim == 1:
        user_samples = np.expand_dims(user_samples,1)
    numVars = np.size(user_samples,1)
    canonical_samples = user_samples.copy()
    for ii in range(numVars):
        loc,scale = scale_parameters[ii,:]
        canonical_samples[:,ii] = ((user_samples[:,ii] - loc)/(np.sqrt(2)*scale))
        # canonical_samples[:,ii] = ((user_samples[:,ii]-loc)/(scale*np.sqrt(2)))

    canonical_samples = np.reshape(canonical_samples,s)
    return canonical_samples

def map_from_canonical_space(canonical_samples, numVars, scale_parameters):
    user_samples = canonical_samples.copy()
    for ii in range(numVars):
        loc,scale = scale_parameters[ii,:]
        user_samples[ii,:] = canonical_samples[ii,:]*scale+loc
    return user_samples


    