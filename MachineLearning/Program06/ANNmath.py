import numpy as np

def concateBias(x):
    ''' Concatenate bias=1 term to input by adding a column of ones to the
    right of the inut data. This is useful for training ANN bias.
    '''
    if x.ndim > 1:
        ones = np.ones( (x.shape[0],1), x.dtype)
    else:
        ones = np.ones(1, x.dtype)
    return np.hstack( [x, ones] )

def sigmoid(x):
    ''' Sigmoid function
    '''
    return 1.0 / ( 1 + np.exp(-x) )

def softMax(x):
    ''' Calculate normalized exponential function for weights and input matrix.
    Returns the normalized exponential probabilities for each classes.
    '''
    pr = np.exp(x) # matrix multiply of x and weight
    if pr.ndim > 1:
        return pr / pr.sum(axis=1)[:,None]
    else:
        return pr / pr.sum()

def crossEntNK(yhat, y):
    ''' Cross Entropy function. Used as the error for multi-class ANN.
    Calculate error across classes for all data points, and return mean err.
    '''
    return np.sum(-y*np.log(yhat), axis=1).mean() # avg error over all points

def getRandomSeq(N):
    seq = np.arange(N)
    np.random.shuffle(seq)
    return seq