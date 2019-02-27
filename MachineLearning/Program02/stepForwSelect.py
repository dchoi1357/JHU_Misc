import numpy as np
from functions import Silhouette, pairwiseDist
from kmeans import kMeans

def SelectBestFeature(dataMat, selected, Nk, dists):
    ''' Select the non-selected features that provides the most improvement to
    Silhouette coefficient.

    Given a matrix of data, a list of selected columns, the number of clusters,
    and a pre-computed distance matrix, the function lopos through all of the
    unselected features, calculating the maximum coefficient for the feature 
    when added to the already selected set. 
    '''
    # get list of index of currently unselected features
    unselect = np.where(~np.isin(np.arange(dataMat.shape[1]),selected))[0]
    bestCoeff = -1-1e-9 # worst possible coefficient value is -1
    for n,j in enumerate(unselect): # loop over unselected features
        testSet = np.hstack([selected,j]) # add curr feature to selected ones
        means,labels = kMeans(dataMat[:,testSet], Nk) # cluster w/ test features
        coeff = Silhouette(dataMat,labels,dists).mean() # mean silhouette coeff
        #print((coeff,bestCoeff))
        if coeff > bestCoeff: # if this feature produce better coeff
            bestCoeff = coeff # record new best coeff
            outs = (j,coeff,means,labels) # record output variables
    #print(unselect)
    return outs # output: the feature, best coeff, means, and labels

def ForwardSelect(data, k, trace=False):
    ''' Step-wise forward selection method
    Start with an empty set of features. Iteratively add the one feature out of
    the non-chosen features which improves the Silhouette coefficient the most. 

    The algorthim is to have converged when adding any feature does not improve
    the coefficint, or no features remain unchosen.
    '''
    selected = np.zeros(0, int) # idx of selected features, start w/ empty
    baseCoeff = -1-1e-9 # -1 is worst possible performance
    dM = pairwiseDist(data) # pre-calc distance matrix for memoization
    
    converged,nRound = False,1
    while not converged: # loop until convergence
        bestFeat,bestCoeff,means,labels = SelectBestFeature(data,selected,k,dM) 
        if bestCoeff <= baseCoeff: # if new feature doesn't improve performance
            converged = True
        else: # if new feature improves performance
            selected = np.hstack([selected,bestFeat]) # add feature to selection
            baseCoeff = bestCoeff # set new coeff as baseline performance
            outs = (means,labels) # save output vars
            if len(selected) == data.shape[1]: 
                converged = True # algo converged if all features selected
        if trace: # print iteration info if requesed
            tmplate = "[%02d] Best coeff=%f, set:%s"
            print( tmplate%(nRound,bestCoeff,str(selected)) )
        nRound += 1
    return (selected,)+outs # return selected features, means, cluster labels