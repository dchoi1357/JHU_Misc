import numpy as np

def concateIntercept(x):
    ''' Concatenate Intercept term to input by adding a column of ones to the
    left of the inut data. This is useful for training regression intercepts.
    '''
    return np.hstack( [np.ones((x.shape[0],1), x.dtype), x] )

def softMax(x, wts):
    ''' Calculate normalized exponential function for weights and input matrix.

    Returns the normalized exponential probabilities for each classes.
    '''
    pr = np.exp(x @ wts) # matrix multiply of x and weight
    return pr / np.sum(pr,axis=1)[:,None]

def fitLogisticNK(x, y, eta, eps=1e-7, trace=False):
    ''' Fit multinomial logistic regression via gradient descent.

    Given an input matrix, a matrix of class labels, and the learning rate, the
    function will iteratively fit a multinomial regression by applying the 
    gradient descent rule. The algorithm stops when the difference between 
    mean errors of successive iterations become smaller than the eps argument,
    or when number of iteration reaches a million

    Returns the learned weight matrix and the number of total iteration taken.
    '''

    def crossEntNK(x, wts, y):
        ''' Cross Entropy function for regression. Used as the error for
        logistic regression. Returns the mean error across all data points and
        the fitted y value based on softmax.
        '''
        yhat = softMax(x, wts) # get normalized exponential probabilities
        err = np.sum(-y*np.log(yhat), axis=1).mean() # avg error over all points
        return err, yhat
    
    def updateWeightNK(eta, yhat, y, x, wt):
        ''' Update the weight matrix given in the argument by applying
        gradient descent rule.
        '''
        d = ((y - yhat).T @ x).T / len(y)
        return wt + (eta*d)
    
    x = concateIntercept(x) # concatenate intercept
    nDim,nK = x.shape[1],y.shape[1] # num of dimensions and cardinality of class
    wts = np.random.rand(nDim,nK)/50 - 0.01 # init wts to be (-0.01,0.01)
    lastErr = np.inf # max error possible
    err,yhat = crossEntNK(x, wts, y) # initial error

    n = 0
    while (abs(err-lastErr) > eps) and n < 1e6: # while not converged
        if n % 1000 == 0 and trace:
            print('Iter #%u, error: %f'%(n,err))
        wts = updateWeightNK(eta, yhat, y, x, wts) # gradient descent upodate
        lastErr = err 
        err,yhat = crossEntNK(x, wts, y) # recalculate error
        if err > lastErr: # slow learning rate if error increase
            eta /= 10
        n += 1
    
    if trace:
        print('Final iteration #%u, error: %f' % (n-1,err) )
    return wts,n

def predLogisticNK(x, wts):
    ''' Predict the class given an input matrix and learned weight.
    '''

    x = concateIntercept(x) # concatenate intercept
    yhat = softMax(x, wts) # calc posterior prob of all classes
    return yhat.argmax(axis=1) # return class with largest probability

####################################
## Single class logistic regression

def sigmoid(x, wt):
    ''' Sigmoid function of matrix multiplication of data and weight vector.
    '''
    return 1.0 / ( 1 + np.exp(-x @ wt) )

def fitLogisticReg(x, y, eta, eps=1e-7, trace=False):
    ''' Fit logistic regression via gradient descent.

    Given an input matrix, a vector of binary class, and the learning rate, the
    function will iteratively fit a logistic regression by applying the 
    gradient descent rule. The algorithm stops when the difference between 
    mean errors of successive iterations become smaller than the eps argument,
    or when number of iteration reaches a million

    Returns the learned weight vector and the number of total iteration taken.
    '''
    m = np.finfo(np.float32).eps # very small value to prevent log(0)

    def crossEntropy(x, wt, y):
        ''' Cross Entropy function for regression. Used as the error for
        logistic regression. Returns the mean error across all data points and
        the fitted y value based on softmax.
        '''
        yhat = sigmoid(x, wt)
        err = -np.mean(y*np.log(yhat+m) + (1-y)*np.log(1-yhat+m))
        return err, yhat
    
    def updateWeight(eta, yhat, y, x, wt):
        ''' Update the weight vector given in the argument by applying
        gradient descent rule. '''
        d = (y - yhat) @ x / len(y)
        return wt + (eta*d)
    
    x = concateIntercept(x) # concatenate intercept
    wt = np.random.rand(x.shape[1])/50 - 0.01 # initialize weights
    lastErr = 1 # max error possible
    err,yhat = crossEntropy(x, wt, y) # initial error

    n = 0
    while (abs(err-lastErr) > eps) and n < 1e6: # while not converged
        if n % 1000 == 0 and trace:
            print('Iter #%u, error: %f'%(n,err))
        wt = updateWeight(eta, yhat, y, x, wt) # gradient descent upodate
        lastErr = err
        err,yhat = crossEntropy(x, wt, y) # recalculate error
        if err > lastErr: # slow learning rate if error increase
            eta /= 10
        n += 1
    
    if trace:
        print('Final iteration #%u, error: %f' % (n-1,err) )
    return wt,n

def predLogistic(x, wt):
    x = concateIntercept(x)
    yhat = sigmoid(x, wt) # calc posterior prob of binary response
    return (yhat > 0.5)*1 # whether posterior prob > 0.5