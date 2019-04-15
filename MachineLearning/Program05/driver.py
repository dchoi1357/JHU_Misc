import sys, os
import pandas as pd

# generate one-hot coding for issues with lots of missing votes
def oneHot(data, colName):
    x = data.loc[:,colName]
    oneHotMat = pd.concat([(x=='y'),(x=='n'),(x=='?')], axis=1)
    oneHotMat.columns = [colName+'_'+suff for suff in ['y','n','q']]
    return oneHotMat

def discretizeMean(inDF, useMed=False):
    outDF = pd.DataFrame()
    if useMed:
        thresh = inDF.median()
    else:
        thresh = inDF.mean()
    for v in list(inDF): # loop over all columns
        outDF[v] = (inDF[v] > thresh[v]) * 1
    return outDF