import numpy as np, pandas as pd
from itertools import product as binCombo # for binary combinations
from kmeans import kMeans
from functions import Silhouette, pairwiseDist

def BruteForceSelect(data, k):
    ''' Try all 2^F-1 combinations of selection, where F = number of features.

    This function tries every possible selection combination to find the one 
    with the best possible Silhouette coefficients.

    Returns the selected columns, the centroids, the cluster assignments, and
    the coefficient of the best set.
    '''
    if data.shape[1] > 15: # error out if no hope of algorithm finishing
        Warning("Too many combinations to try.")
        return
    
    # get all binary combination of features (e.g. whether to include)
    combos = np.array( list(binCombo([True,False],repeat=data.shape[1])) )
    combos = combos[(combos==True).any(axis=1)] # remove combo w/ no features
    
    dist = pairwiseDist(data) # pre-calc distance matrix for memoization
    coeffs = np.empty(combos.shape[0]) # store Silhouette coeff of combos
    means = [None]*combos.shape[0] # store centroids of all combos
    groups = [None]*combos.shape[0] # to store labels of all combinations
    
    for n,featIdx in enumerate(combos):
        means[n],groups[n] = kMeans(data[:,featIdx], k) # cluster w/ features
        coeffs[n] = Silhouette(data,groups[n],dist).mean() # mean coeffs
    
    idx = np.argmax(coeffs) 
    return np.where(combos[idx])[0],means[idx],groups[idx],coeffs[idx]