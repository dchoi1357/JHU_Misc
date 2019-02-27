import numpy as np
from functions import pairwiseDist

def initMeans(data, n, algo=1):
    ''' Initialize centroids for K-Means algorithm
    To ensure reproducibility, the function uses an independent random stream
    with a set seed so that initialization of means are deterministic.

    There are three possible algorithms, 1) choose random points in data, 
    2) choose random points between (0,1), 3) always take the top K data points.
    Returns the initialized centroids
    '''
    gen = np.random.RandomState() # use independent stream for reproducibility
    gen.seed(42) # set initial seed
    
    if algo==1: # choose n random points
        idx = gen.choice(range(data.shape[0]), n, False) # no replace
        out = data[idx,]
    if algo==2: # random means points (0,1)
        tmp = [[gen.rand() for y in range(data.shape[1])] for x in range(n)]
        out = np.array(tmp)
    if algo==3: # always take first n point as centroid
        out = data[:n,]
    return out


def updateMeans(data, means):
    ''' Calculate and update centroids for K-means algorithm. 
    The function has two parts:
    1) Assign each pt to the mean for which it has the shortest distance
    2) Calculate new means to be centroid of all the points in the group
    Returns the new means and the classification of data points to these means
    '''
    tmpDist = pairwiseDist(means,data) # dist between means and all data pts
    minClus = tmpDist.argmin(axis=0) # find group where distance is smallest

    newMeans = np.zeros([len(means),data.shape[1]]) # new mean points
    for n,x in enumerate(means): # loop over all clusters
        tmp = np.vstack( (data[minClus==n,],x) ) # concat data pt and centroid
        newMeans[n] = tmp.mean(axis=0) # new mean = centroid of all pts 
    
    return newMeans,minClus


def kMeans(data, k, trace=False, initAlgo=1):
    ''' Main function of K-means algorithm
    Performs k-means clustering on the data for the number of clusters as 
    specified in the input parameter _k_. Algorithm stops when successive 
    centroids are close to each others in all the dimensions. Returns the final
    centroids and the cluster classification for all data points.
    '''
    means = initMeans(data, k, initAlgo) # initialize mean points
    converged = False
    while not converged:
        newMeans,grpIdx = updateMeans(data, means)
        converged = np.allclose(means,newMeans)
        if trace:
            print(means)
        means = newMeans
        
    return means,grpIdx # return final centroids and labels