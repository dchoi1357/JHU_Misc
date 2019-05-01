import numpy as np
from ANNmath import concateBias, sigmoid, softMax, crossEntNK, getRandomSeq

def train_2hidd(xMat, yMat, eta, nNodes, eps=1e-6, trace=False, shuffle=True):
    def feedForward(xs, ys, wtsOut, wtsHidd2, wtsHidd1):
        z1s = concateBias( sigmoid(xs@wtsHidd1) )
        z2s = concateBias( sigmoid(z1s@wtsHidd2) )
        return (z1s,z2s), softMax(z2s @ wtsOut)
    
    def backProp(ys, yfit, xs, zs, wtsOut, wtsHidd2, wtsHidd1):
        z1s,z2s = zs
        errO = ys-yfit
        d_Out = eta * np.outer(z2s, errO)
        
        err2 = (wtsOut@errO) * (z2s*(1-z2s))
        d_hidd2 = eta * np.outer(z1s,err2)[:,:-1]
        
        err1 = (wtsHidd2@err2[:-1]) * (z1s*(1-z1s))
        d_hidd1 = eta * np.outer(xs,err1)[:,:-1]
        return wtsOut + d_Out, wtsHidd2 + d_hidd2, wtsHidd1 + d_hidd1
    
    xMat = concateBias(xMat)
    (nData,nK),nDim = yMat.shape, xMat.shape[1]
    
    wtOut = np.random.rand(nNodes+1,nK)/50 - 0.01 # init wts to be (-0.01,0.01)
    wtHidd2 = np.random.rand(nNodes+1,nNodes)/50 - 0.01
    wtHidd1 = np.random.rand(nDim,nNodes)/50 - 0.01
    
    lastErr = np.inf # max error possible
    zs,yHats = feedForward(xMat, yMat, wtOut, wtHidd2, wtHidd1)
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
            z12,yHat = feedForward(x, y, wtOut, wtHidd2, wtHidd1) # feedforward
            wtOut,wtHidd2,wtHidd1 = backProp(y, yHat, x, z12, # update wts
                                             wtOut, wtHidd2, wtHidd1) 

        lastErr = meanErr        # fitted Y for this epoch
        zs,yHats = feedForward(xMat, yMat, wtOut, wtHidd2, wtHidd1) 
        meanErr = crossEntNK(yHats, yMat) # err for this epoch
        
        if meanErr > lastErr:  # slow learning rate if error increase
            eta /= 2
        epch += 1

    if trace: # print final error
        print('Final iteration #%u, error: %f' % (epch-1,meanErr) )
    return (wtOut,wtHidd2,wtHidd1),epch,meanErr

def pred_2hidd(xMat, wtsOut, wtsHidd2, wtsHidd1):
    z1 = sigmoid(concateBias(xMat) @ wtsHidd1)
    z2 = sigmoid( concateBias(z1) @ wtsHidd2 )
    yHat = softMax(concateBias(z2) @ wtsOut)
    return yHat.argmax(axis=1)