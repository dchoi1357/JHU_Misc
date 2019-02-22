import numpy as np, pandas as pd

def WinnowTrain(data, classVec, prm, trace=False):
    wts = np.ones(data.shape[1]) # initialize weight vector
    if trace: # print initial weights if trace is on
        print('initial weights: %s'%wts) 
    for n,x in enumerate(data):
        pred = wts.dot(x) > prm['theta'] # prediction is if f > theta
        if pred != classVec[n]: # wrong prediction: promotion / demotion
            mult = (pred==0)*prm['alpha'] + (pred==1)/prm['alpha'] 
            wts = (x==1)*wts*mult + (x==0)*wts # update weights
            if trace: # print updated weights if its 
                print('[%d] new weights: %s'%(n,wts)) 
        else: # correct prediction, no update needed, only for tracing
            if trace: # print that no update to weights
                print('[%d] no update to weights'%n)
    if trace: # print final weights for trace
        print('Final weights: %s'%wts)
    return wts

def WinnowPred(data, wts, prm):
    return (data.dot(wts) > prm['theta'])*1 # prediction: if f > theta

def errRates(pred, actual):
    return np.sum(actual!=pred)/pred.size # return error rate