import numpy as np
from ANNmath import concateBias, softMax, crossEntNK, getRandomSeq

def train_0hidd(xMat, yMat, eta, eps=1e-6, trace=False, shuffle=True):
    def feedForward(xs, ys, wts):
        return softMax(xs @ wts)
    
    def backProp(ys, yfit, xs, wts):
        return wts + eta * np.outer(xs, ys-yfit)
    
    xMat = concateBias(xMat) # add bias terms
    (nData,nK),nDim = yMat.shape, xMat.shape[1] # size of data and classes
    
    wt = np.random.rand(nDim,nK)/50 - 0.01 # init wts to be (-0.01,0.01)
    lastErr = np.inf # max error possible
    yHats = feedForward(xMat, yMat, wt) # first feedforward calc
    meanErr = crossEntNK(yHats, yMat) # error from random weights
    
    epch = 0
    while (abs(meanErr-lastErr) > eps) and epch < 1e6: # while not converged
        if epch%1000==0 and trace:
            print('Iter #%u, error: %f'%(epch,meanErr))
        
        if shuffle: # shuffle sequence of gradient descent
            seq = getRandomSeq(nData) # random seq for stoch. gradient descent
        else:
            seq = np.arange(nData)
        for n in seq: # loop over data set
            x,y = xMat[n],yMat[n] # index x and y for curr data point
            yHat = feedForward(x, y, wt) # feedforward
            wt = backProp(y, yHat, x, wt) # update weight
        
        lastErr = meanErr
        yHats = feedForward(xMat, yMat, wt) # fitted Y for this epoch
        meanErr = crossEntNK(yHats, yMat) # err for this epoch
        
        if meanErr > lastErr:  # slow learning rate if error increase
            eta /= 5
        epch += 1

    if trace: # print final error
        print('Final iteration #%u, error: %f' % (epch-1,meanErr) )
    return wt,epch,meanErr

def pred_0hidd(xMat, wts):
    yHat = softMax(concateBias(xMat) @ wts)
    return yHat.argmax(axis=1)