import numpy as np
import LogisticRegression as LG
import NaiveBayes as NB

def errRate(pred, actual, categorical=True):
    if categorical: # if categ., return classification err rate
        return sum(pred!=actual) / pred.size
    else: # if numeric, return RMSE
        return np.linalg.norm(pred-actual)/np.sqrt(pred.size)

def getXVFolds(dataMat, classVec, nFolds=5, categorical=True):
    ''' Cut N-fold cross validation of the data set
    Given a data matrix, a class vector, and the number of folds, the function
    randomly cuts a 5-fold cross validation. If the data is categorical, 
    stratified sampling is used.
    '''

    idx = np.arange(dataMat.shape[0]) # construct index of data rows
    if categorical:
        if classVec.ndim > 1: # if labels have more than one dimension
            classVec = classVec.argmax(axis=1) # collapse to 1D 
            
        unqs = np.unique(classVec)
        tmpHold = [None] * len(unqs)
        for n,k in enumerate(unqs):
            grpIdx = idx[classVec==k] # idx of all elems in current class
            np.random.shuffle(grpIdx) # permutate idx for random selection
            tmpHold[n] = np.array_split(grpIdx, nFolds) # split: N equals
        chunks = [np.hstack(k) for k in zip(*tmpHold)] # concat sub chunks
    else:
        np.random.shuffle(idx) # random shuffle data
        chunks = np.array_split(idx, nFolds) # split into N equal sized chunks

    return chunks # return the prediction of the last fold


def crossValidate(data, labels, chunks, dataCont=None):
    ''' Perform n-fold cross validation.
    Given data array, labels, the chunks as folds, the function performs
    cross validation by using 4 out of 5 folds as training and the fifth as 
    testing set. This is repeated 5 times with a different chunk of the fold 
    serving as testing set. With each fold, both NB and logistic regression 
    is fitted. The validation error with both algorithms are returned.
    '''

    errLG = np.empty(len(chunks)) # pred error, logistic regression
    nIters = np.empty(len(chunks), int) # number of iterations for logistic reg
    errNB = np.empty(len(chunks)) # pred error, naive Bayes
    
    for ck in range(len(chunks)):
        # get index and dataset for current fold of cross-validation
        trnIdx = np.hstack([x for n,x in enumerate(chunks) if n != ck])
        vldIdx = np.hstack([x for n,x in enumerate(chunks) if n == ck])
        dataTrain,labelTrain = data[trnIdx],labels[trnIdx] # training
        dataTest,labelTest = data[vldIdx],labels[vldIdx] # validation
        
        if labels.ndim > 1: # if labels have more than one dimension
            labelTest = labelTest.argmax(axis=1) # collapse to 1D 

        ## Fit and predict with naive Bayes
        prb = NB.NB_Train(dataTrain, labelTrain)
        predNB = NB.NB_Pred(dataTest, prb)
        errNB[ck] = errRate(predNB, labelTest) # error with naive Bayes

        ## Fit and predict with logistic regression
        if dataCont is not None: # if given non-discretized data
            dataTrain = dataCont[trnIdx]
            dataTest = dataCont[vldIdx]
        if labels.ndim > 1: # if more than two classes, use multinomial logistic
            wts,nIters[ck] = LG.fitLogisticNK(dataTrain, labelTrain, 0.5)
            predLG = LG.predLogisticNK(dataTest, wts)
        else: # binary response var, use regular logistic regression
            wt,nIters[ck] = LG.fitLogisticReg(dataTrain, labelTrain, 0.5)
            predLG = LG.predLogistic(dataTest, wt)
        errLG[ck] = errRate(predLG, labelTest) # error with logistic reg

    return errLG,errNB,nIters