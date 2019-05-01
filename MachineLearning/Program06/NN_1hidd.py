import numpy as np
from ANNmath import concateBias, sigmoid, softMax, crossEntNK, getRandomSeq

def train_1hidd(xMat, yMat, eta, nNodes, eps=1e-6, trace=False, shuffle=True):
    def feedForward(xs, ys, wtsOut, wtsHidd):
        zs = concateBias( sigmoid(xs@wtsHidd) )
        return zs, softMax(zs @ wtsOut)
    
    def backProp(ys, yfit, xs, zs, wtsOut, wtsHidd):
        d_Out = eta * np.outer(zs, ys-yfit)
        d_hidd = eta * np.outer(xs, wtsOut@(ys-yfit) * (zs*(1-zs)))[:,:-1]
        return wtsOut + d_Out, wtsHidd + d_hidd
    
    xMat = concateBias(xMat)
    (nData,nK),nDim = yMat.shape, xMat.shape[1]
    
    wtOut = np.random.rand(nNodes+1,nK)/50 - 0.01 # init wts to be (-0.01,0.01)
    wtHidd = np.random.rand(nDim,nNodes)/50 - 0.01
    
    lastErr = np.inf # max error possible
    zs,yHats = feedForward(xMat, yMat, wtOut, wtHidd)
    meanErr = crossEntNK(yHats, yMat)
    
    epch = 0
    while (abs(meanErr-lastErr) > eps) and epch < 1e6: # while not converged
        if epch%1000==0 and trace:
            print('Iter #%u, error: %f'%(epch,meanErr))
        
        if shuffle:
            seq = getRandomSeq(nData) # random seq for stoch. gradient descent
        else:
            seq = np.arange(nData)
        for n in seq: # loop over data set
            x,y = xMat[n],yMat[n] # index x and y for curr data point
            z,yHat = feedForward(x, y, wtOut, wtHidd) # feedforward
            wtOut,wtHidd = backProp(y, yHat, x, z, wtOut, wtHidd) # update weight

        lastErr = meanErr
        zs,yHats = feedForward(xMat, yMat, wtOut, wtHidd) # fitted Y for this epoch
        meanErr = crossEntNK(yHats, yMat) # err for this epoch
        
        if meanErr > lastErr:  # slow learning rate if error increase
            eta /= 2
        epch += 1

    if trace: # print final error
        print('Final iteration #%u, error: %f' % (epch-1,meanErr) )
    return (wtOut,wtHidd),epch,meanErr

def pred_1hidd(xMat, wtsOut, wtsHidd):
    z = sigmoid(concateBias(xMat) @ wtsHidd)
    yHat = softMax(concateBias(z) @ wtsOut)
    return yHat.argmax(axis=1)