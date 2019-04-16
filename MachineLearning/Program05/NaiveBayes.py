import numpy as np, pandas as pd

def condProb(data, add): # data assumed to be class-homogenous
    ''' Calculate the conditional probability of a class-homogenous data set.
    The function returns the conditional probability with Laplace smoothing. 
    Data matrix has to be binary 0-1.
    '''
    condPr = np.zeros(data.shape[1]) # pre-allocate cond probilities
    for n,x in enumerate(data.T): # loop over the columns of the data
        condPr[n] = (sum(x==1)+add)/(len(x)+add) # laplace smooth if needed
    return condPr

def NB_Train(data, classVec, smooth=True):
    ''' Trains Naive Bayes on an input data matrix and class label. 
    If smooth = True, then Laplace smoothing is performed.

    Returns 3-tuple of probabilities, cond prob of C=0, cond prob of C=1
    '''
    smoothAdd = smooth*1 # addition to num and denom for smoothing
    if classVec.ndim==1: # binary class vector
        classVec = np.hstack([classVec==1,classVec==0])
        
    pr_class = np.empty(classVec.shape[1], float) # probability of classes
    condPrs = list()
    for n,vec in enumerate(classVec.T): # loop over classes
        idx = (vec==1) # all data points belonging to this class
        condPrs.append( condProb(data[idx],smoothAdd) ) # calc cond probs
    
    return (pr_class,condPrs) # return class prob and cond probs

def NB_Pred(data, probs): # predicting based on conditional probs
    PrX = np.empty([len(data), len(probs[0])], float)
    for n,(pr,cond) in enumerate(zip(*probs)): # loop over classes
        PrX[n] = np.log(pr) # uncond probability
        tmp = (data==1)*cond + (data==0)*(1-cond) # cond prob in class
        PrX[n] += np.log(tmp).sum(axis=1) # sum log probabilities
    
    return PrX.argmax(axis=1) # return most likely classification
