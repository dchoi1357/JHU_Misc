import numpy as np, pandas as pd

def condProb(data, add): # data assumed to be class-homogenous
    ''' Calculate the conditional probability of a class-homogenous data set.
    The function returns the conditional probability with Laplace smoothing. 
    Data matrix has to be binary 0-1.
    '''
    condPr = np.zeros(data.shape[1]) # pre-allocate cond probilities
    for n,x in enumerate(data.T): # loop over the columns of the data
        condPr[n] = (sum(x==0)+add)/(len(x)+add) # laplace smooth if needed
    return condPr

def NB_Train(data, classVec, smooth=True):
    ''' Trains Naive Bayes on an input data matrix and class label. 
    If smooth = True, then Laplace smoothing is performed.

    Returns 3-tuple of probabilities, cond prob of C=0, cond prob of C=1
    '''
    smoothAdd = smooth*1 # addition to num and denom for smoothing
    
    pr_C0 = sum(classVec==0)/len(classVec) # probability of class=0
    condPr_C0 = condProb(data[classVec==0,], smoothAdd) # cond prob class=0
    condPr_C1 = condProb(data[classVec==1,], smoothAdd) # cond prob class=1
    return (pr_C0,condPr_C0,condPr_C1) # return class prob and cond probs

def NB_pred(data, probs): # predicting based on conditional probs
    pr_C0,condPr0_C0,condPr0_C1 = probs
    xCondsC0 = (data==0)*condPr0_C0 + (data==1)*(1-condPr0_C0)
    xCondsC1 = (data==0)*condPr0_C1 + (data==1)*(1-condPr0_C1)
    PrX_C0 = np.cumprod(xCondsC0,1)[:,-1] * pr_C0
    PrX_C1 = np.cumprod(xCondsC1,1)[:,-1] * (1-pr_C0)
    return (PrX_C1>PrX_C0)*1