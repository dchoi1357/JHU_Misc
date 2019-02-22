import numpy as np, pandas as pd
import functions.naiveBayes as NB
import functions.winnow as WN

def errRates(pred, actual):
    return np.sum(actual!=pred)/pred.size # return error rate

def crossValidate(dataMat, classVec, hyPrm, trace=False):
    ''' Perform 10-fold cross validation of the data set
    First, randomly shuffle and split the input data set into 10. For every
    fold, perform both estimation and validation of both Winnow and Naive Bayes.
    Afterwards, print out the error rate and standard deviation of both.
    Returns the data indices and predicted class of the validation set for the 
    last fold.
    '''
    slices = 10
    idx = np.arange(dataMat.shape[0]) # construct index of data rows
    np.random.shuffle(idx) # random shuffle data
    chunks = np.array_split(idx, slices) # split into N equal sized chunks

    errsWinnow = np.zeros(slices)  # pre-allocate Winnow errors for each fold
    errsNB = np.zeros(slices) # pre-allocate NB errors for each fold
    for n in range(slices): # loop over all slices
        # get index and dataset for current fold of cross-validation
        trnIdx = np.hstack([chunks[x] for x in range(slices) if x != n])
        vldIdx = np.hstack([chunks[x] for x in range(slices) if x == n])
        dataTrain,classTrain = dataMat[trnIdx,:],classVec[trnIdx] # training
        dataVald,classVald = dataMat[vldIdx,:],classVec[vldIdx] # validation

        # train and test Winnow algorithm
        wts = WN.WinnowTrain(dataTrain, classTrain, hyPrm, False)
        pred = WN.WinnowPred(dataVald, wts, hyPrm)
        errsWinnow[n] = errRates(pred, classVald)
        
        # train and test Naive Bayes
        probs = NB.NB_Train(dataTrain, classTrain, smooth=True)
        pred = NB.NB_pred(dataVald, probs)
        errsNB[n] = errRates(pred, classVald)

    print('Average error rate for Winnow is %f.'%np.mean(errsWinnow))
    print('Std Dev of error rate for Winnow is %f.'%np.std(errsWinnow))
    print('Average error rate for NB is %f.'%np.mean(errsNB))
    print('Std Dev of error rate for NB is %f.'%np.std(errsNB))

    out = pd.DataFrame() # prepare data frame to return prediction of 10th fold
    out["id"] = vldIdx
    out['pred'] = pred
    return out # return the prediction of the last fold

def estModels(dataMat, classVec, hyPrm, trace=False):
    ''' Estimate both Winnow and Naive Bayes model based on input data and
    return the model parameters back to the user in a tuple '''
    wts = WN.WinnowTrain(dataMat, classVec, hyPrm, False)
    probs = NB.NB_Train(dataMat, classVec, smooth=True)
    return wts,probs

def printWinnowModel(wts, colNames):
    '''Print the weights of the Winnow model in a human understandable
    fashion. Require the weights and column names as input.
    '''
    toPrint = pd.DataFrame()
    toPrint["features"] = colNames
    toPrint["weights"] = wts
    print("Winnow-2 Model Weights:")
    print(toPrint)
    print()
    pass

def printNBmodel(probs, colNames):
    '''Print the Naive Bayes probabilities in a human understandable
    fashion. Require the a three-tuple of:
        - unconditional C=0 probability
        - conditional feature probabilities given C=0
        - conditional feature probabilities given C=1
    '''
    unconds = pd.DataFrame()
    unconds['Class'] = [0,1]
    unconds['Uncond Pr'] = [probs[0],1-probs[0]]
    print("NB Probabilities of Classes:")
    print(unconds)
    
    print("\nNB Conditional Probabilities of Features:")
    conds = pd.DataFrame()
    conds['features'] = colNames
    conds['[C=0,X=0]'] = probs[1]
    conds['[C=0,X=0]'] = 1-probs[1]
    conds['[C=1,X=0]'] = probs[2]
    conds['[C=1,X=1]'] = 1-probs[2]
    print(conds)