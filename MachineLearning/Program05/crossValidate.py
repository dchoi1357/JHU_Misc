import numpy as np

def getXVFolds(dataMat, classVec, nFolds=5, categorical=True):
    ''' Cut N-fold cross validation of the data set
    Given a data matrix, a class vector, and the number of folds, the function
    randomly cuts a 5-fold cross validation. If the data is categorical, 
    stratified sampling is used.
    '''

    idx = np.arange(dataMat.shape[0]) # construct index of data rows
    if categorical:
        unqs = np.unique(classVec)
        tmpHold = [None] * len(unqs)
        for n,k in enumerate(unqs):
            grpIdx = idx[classVec==k] # idx of all elems in current class
            np.random.shuffle(grpIdx) # permutate idx for random selection
            tmpHold[n] = np.array_split(grpIdx, nFolds) # split: N equals
        chunks = [np.hstack(k) for k in zip(*tmpHold)] # concat sub chunks
    else:
        np.random.shuffle(idx) # random shuffle data
        chunks = np.array_split(idx, nFolds) # split into N equal sized chunks

    return chunks # return the prediction of the last fold