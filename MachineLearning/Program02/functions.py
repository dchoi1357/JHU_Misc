import numpy as np, pandas as pd

def prepData(dataPathDir, fieldNames, featSlices):
    raw = pd.read_csv(dataPathDir , names=fieldNames)  # read CSV file
    dataFeats = fieldNames[featSlices] # list of feature names
    meanVals = raw[dataFeats].mean().values # mean of all features
    #dataMat = raw[dataFeats].values/meanVals # standardized array of vals
    dataMat = raw[dataFeats].values # standardized array of vals
    nK = len(raw['class'].unique()) # number of classes
    return dataMat,dataFeats,nK,meanVals

def pairwiseDist(x, y=None):
    if y is None:
        y = x
    return np.sum((x[:,None]-y)**2,axis=2)**0.5

def Silhouette(data, labels, distMat=None):
    if distMat is None:
        distMat = pairwiseDist(data) # calc pairwise dist if not provided
    grpIdx = pd.Series(labels).groupby(labels).groups.items() # idx for each grp
    
    aVals = np.zeros(data.shape[0]) # pre-allocate a and b-values for data
    bVals = np.zeros(data.shape[0])
    for grp,idx in grpIdx: # loop over all groups
        aVals[idx] = distMat[np.ix_(idx,idx)].mean(axis=1) # a's for curr grp
        
        # loop over all groups that's not the current gruop
        tmp = np.zeros([len(grpIdx)-1,len(idx)]) # tmp for all b's for curr grp
        for n,(_,outIdx) in enumerate([x for x in grpIdx if x[0]!=grp]):
            # calculate mean dist of points within cluster to out of cluster
            tmp[n,] = distMat[np.ix_(idx,outIdx)].mean(axis=1) 
        bVals[idx] = tmp.min(axis=0) # pick min b of all out-groups

    return (bVals-aVals)/np.maximum(aVals,bVals) # return silhouette coeff

def printResults(outputLoad, featNames):
    selected,centroids,labels = outputLoad[:3]
    counts = np.zeros(len(centroids))
    for cl in labels:
        counts[cl] += 1
    
    out = pd.DataFrame(np.vstack([centroids.T,counts]))
    selectedFeats = [featNames[n] for n in selected] + ['Counts']
    out = out.rename(index={n:x for n,x in enumerate(selectedFeats)})
    print("Features and cluster centroids")
    print(out)
