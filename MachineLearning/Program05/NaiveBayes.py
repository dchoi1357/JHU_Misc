import numpy as np, pandas as pd


def condProb(data, add): # data assumed to be class-homogenous
    ''' Calculate the conditional probability of a class-homogenous data set.
    The function returns the conditional probability with Laplace smoothing. 
    Data matrix has to be binary 0-1.
    '''
    condPr = np.zeros(data.shape[1]) # pre-allocate cond probilities
    for n,x in enumerate(data.T): # loop over the columns of the data
        condPr[n] = (sum(x==0)+add)/(len(x)+add*2) # laplace smooth if needed
    return condPr

def NB_Train(data, classVec, smooth=True):
    ''' Trains Naive Bayes on an input data matrix and class label. 
    If smooth = True, then Laplace smoothing is performed.

    Returns tuple of uncond class probs, cond prob of features for each class
    '''
    smoothAdd = smooth*1 # addition to num and denom for smoothing
    if classVec.ndim==1: # binary class vector, transform into 2D
        classVec = np.vstack([classVec==0,classVec==1]).T *1 # [0's, 1's]

    pr_class = np.empty(classVec.shape[1], float) # probability of classes
    condPrs = list()
    for n,vec in enumerate(classVec.T): # loop over classes
        idx = (vec==1) # all data points belonging to this class
        pr_class[n] = sum(idx) / len(idx) # uncond probability
        condPrs.append( condProb(data[idx],smoothAdd) ) # calc cond probs
    
    return (pr_class,condPrs) # return class prob and cond probs

def NB_Pred(data, probs): # predicting based on conditional probs
    ''' Predict naive Bayes based on unconditional class probability and 
    conditional feature probability of each class, as well as the data matrix.

    Return the predicted class based on naive Bayes.
    '''
    PrX = np.empty([len(data), len(probs[0])], float) # T-by-K matrix
    for n,(pr,cond) in enumerate(zip(*probs)): # loop over classes
        tmp = (data==0)*cond + (data==1)*(1-cond) # cond prob in class
        #PrX[:,n] = pr * tmp.prod(axis=1) # prod of uncond and cond probs
        PrX[:,n] = np.log(pr) + np.log(tmp).sum(axis=1) # sum log probabilities

    return PrX.argmax(axis=1) # return most likely classification