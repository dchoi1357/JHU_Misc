import numpy as np
from functions import Silhouette
from kmeans import kMeans

def SelectBestFeature(dataMat, selected, Nk):
    # get list of index of currently unselected features
    unselect = np.where(~np.isin(np.arange(dataMat.shape[1]),selected))[0]
    bestCoeff = -1 # worst possible coefficient value
    for n,j in enumerate(unselect): # loop over unselected features
        testSet = np.hstack([selected,j]) # add curr feature to selected ones
        means,labels = kMeans(dataMat[:,testSet], Nk) # cluster w/ test features
        coeff = Silhouette(dataMat,labels).mean() # mean silhouette coeff
        if coeff > bestCoeff: # if this feature produce better coeff
            bestCoeff = coeff # record new best coeff
            outs = (j,coeff,means,labels) # record output variables
    return outs # output: the feature, best coeff, means, and labels


def ForwardSelect(data, k):
    selected = np.empty(0, int) # idx of selected features, start w/ empty
    baseCoeff = -1 # -1 is worst possible performance
    converged = False
    while not converged: # loop until convergence
        bestFeat,bestCoeff,means,labels = SelectBestFeature(data, selected, k) 
        if bestCoeff <= baseCoeff: # if new feature doesn't improve performance
            converged = True
        else: # if new feature improves performance
            selected = np.hstack([selected,bestFeat]) # add feature to selection
            baseCoeff = bestCoeff # set new coeff as baseline performance
            outs = (means,labels) # save output vars
    return (selected,)+outs # return selected features, means, cluster labels
