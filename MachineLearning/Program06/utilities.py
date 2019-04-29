import pandas as pd, numpy as np

def discretizeMean(inDF, useMed=False):
    ''' For the input data frame, which is assumed to be continuous, returns
    an output data frame where each feature is replaced by a feature where the 
    x=1 if x > mean(x). 
    '''
    outDF = pd.DataFrame()
    if useMed:
        thresh = inDF.median()
    else:
        thresh = inDF.mean()
    for v in list(inDF): # loop over all columns
        outDF[v] = (inDF[v] > thresh[v])
    return outDF.values * 1


def oneHot(data, colNames):
    ''' Apply one-hot coding to data features.

    For each feature in input, produce one-hot coding by producing vectors of
    1's and 0's for all unique value within each vector.
    '''
    outDF = pd.DataFrame()
    for col in colNames:
        x = data[col]
        for val in x.unique():
            suff = 'q' if val=='?' else str(val)
            outDF[col+'_'+suff] = (x==val)
    return outDF

def normalizeDF(data):
    ''' Normalize the input data frame.

    returns x-min(x) / (range(x)), so all data falls between [0,1]
    '''
    mins = data.min() # min of every col
    maxs = data.max() # max of every col
    return ((data-mins) / (maxs-mins)).values # normalize to [0,1]

def makeClassMat(classVec):
    classes = classVec.unique()
    return pd.concat([classVec==x for x in classes], axis=1).values * 1