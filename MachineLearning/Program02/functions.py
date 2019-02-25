import numpy as np, pandas as pd

def pairwiseDist(x, y=None):
    if y is None:
        y = x
    return np.sum((x[:,None]-y)**2,axis=2)**0.5

def Silhouette(data, labels):
    allDists = pairwiseDist(data) # pre-calc all pairwise dist for memoization
    grpIdx = pd.Series(labels).groupby(labels).groups.items() # idx for each grp
    
    aVals = np.empty(data.shape[0]) # pre-allocate a and b-values for data
    bVals = np.empty(data.shape[0])
    for grp,idx in grpIdx: # loop over all groups
        aVals[idx] = allDists[np.ix_(idx,idx)].sum(axis=1)/len(idx) # tmp a's
        
        # loop over all groups that's not the current gruop
        tmp = np.empty([len(grpIdx)-1,len(idx)]) # tmp for all b's for curr grp
        for n,(_,outIdx) in enumerate([x for x in grpIdx if x[0]!=grp]):
            # calculate mean dist of points within cluster to out of cluster
            tmp[n,] = allDists[np.ix_(idx,outIdx)].mean(axis=1) 
        bVals[idx] = np.min(tmp, axis=1)

    return (bVals-aVals)/np.maximum(aVals,bVals) # return silhouette coeff