import numpy as np
from crossValidation import getXVFolds

def prepData(dataPathDir, fieldNames, featSlices, labelName, sep=','):
    raw = np.genfromtxt(dataPathDir, delimiter=sep, dtype=None,
                        names=fieldNames, encoding='utf-8') # read dlm file
    if isinstance(featSlices, slice):
        dataFeats = fieldNames[featSlices] # list of feature names
    else:
        dataFeats = [fieldNames[i] for i in featSlices]
    return raw[dataFeats],dataFeats,raw[labelName]

def splitData(data, labels):
    x = getXVFolds(data, labels, nFolds=10, categorical=True)
    pruneData,pruneLabel = data[x[0]], labels[x[0]]
    xvSetIdx = np.hstack(x[1:]) # rest of the data
    xvData,xvLabel = data[xvSetIdx], labels[xvSetIdx]
    
    xvFolds = getXVFolds(xvData, xvLabel, categorical=True)
    return (xvData,xvLabel),(xvFolds),(pruneData,pruneLabel)
	


def IntInfo(counts):
    s = sum(counts)
    return -np.sum(np.log2(counts)*counts)/s + np.log2(s)

