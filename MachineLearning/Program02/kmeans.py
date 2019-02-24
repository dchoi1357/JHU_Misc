import numpy as np
from functions import pairwiseDist

def initMeans(data, n, algo=1):
    if algo==1: # choose n random points
        idx = np.random.choice(range(data.shape[0]), n, False) # no replace
        out = data[idx,]
    if algo==2: # random means points (0,1)
        out = np.random.random([n,data.shape[1]])
    return out


def updateMeans(data, means):
    ## Assign each pt to the mean for which it has the shortest distance
    tmpDist = pairwiseDist(means,data) # dist between means and all data pts
    minDist = tmpDist.argmin(axis=0) # find group where distance is smallest

    ## Calculate new means to be centroid of all the points in the group
    newMeans = np.empty([len(means),data.shape[1]]) # new mean points
    for n,x in enumerate(means): # loop over all clusters
        tmp = np.vstack( (data[minDist==n,],x) ) # concat data pt and centroid
        newMeans[n] = tmp.mean(axis=0) # new mean = centroid of all pts 
    
    return newMeans,minDist


def kMeans(data, k, trace=False, initAlgo=1):
    means = initMeans(data, k, initAlgo) # initialize mean points
    converged = False
    while not converged:
        newMeans,grpIdx = updateMeans(data, means)
        converged = np.allclose(means,newMeans)
        if trace:
            print(means)
        means = newMeans
        
    return means,grpIdx # return final centroids and labels