import numpy as np
from utilities import errRate
from NN_0hidd import train_0hidd, pred_0hidd
from NN_1hidd import train_1hidd, pred_1hidd
from NN_2hidd import train_2hidd, pred_2hidd

eta = 2.5 # learning rate

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

def tuneNHiddNodes(dataMat, classMat, folds, rand):
    ''' Find the optimal number of hidden nodes for use with ANN.
    For ANN with 1 and 2 hidden layers, choose 1/5 of the data as testing set, 
    with the rest as training set. For each structure, find the optimal number 
    of nodes in hidden layers by trying every number between nDimensions and 
    nClasses. If there are more than two numbers between those, try every other
    numbers. Return the nHidden nodes that produce the smallest amount of cross
    validation error.
    '''
    
    pick = np.random.randint(len(folds)) # randomly pick fold as validation set
    trnIdx = np.hstack([x for n,x in enumerate(folds) if n != pick]) # train
    vldIdx = np.hstack([x for n,x in enumerate(folds) if n == pick]) # validate
    dataTrain,labelTrain = dataMat[trnIdx,:],classMat[trnIdx] # training
    dataTest,labelTest = dataMat[vldIdx,:],classMat[vldIdx] # validation

    if classMat.ndim > 1: # if labels have more than one dimension
        labelTest = labelTest.argmax(axis=1) # collapse to 1D 

    nDim = dataMat.shape[1] # dimension of data
    nClass = np.unique(classMat).size # number of sizes

    if nDim - nClass +1 > 10: # if candidate larger than 10
        hiddNodes = np.arange(nDim, nClass-1, -2) # try every 2 numbers
    else:
        hiddNodes = np.arange(nDim, nClass-1, -1) # try every num between

    hiddNodes = np.arange(nDim-2, nDim+8) # candidates for num of hidden nodes
    nHiddErr = np.empty([2,hiddNodes.size]) # store error for tuning

    for n,nNode in enumerate(hiddNodes): # loop over candidate nNodes
        wt1,nIters,errIS = train_1hidd(dataTrain, labelTrain, eta, 
                                       nNode, shuffle=rand)
        nHiddErr[0,n] = errRate(pred_1hidd(dataTest,*wt1), labelTest)

        # Fit and predict with ANN with 2 hidden layers
        wt2,nIters,errIS = train_2hidd(dataTrain, labelTrain, eta, 
                                       nNode, shuffle=rand)
        nHiddErr[1,n] = errRate(pred_2hidd(dataTest,*wt2), labelTest)

    return hiddNodes[nHiddErr.argmin(axis=1)] # return nNode w/ lowest err

def doCrossValidate(data, labels, chunks, nHiddNd, rand):
    ''' Perform n-fold cross validation.
    Given data array, labels, the chunks as folds, the function performs
    cross validation by using 4 out of 5 folds as training and the fifth as 
    testing set. This is repeated 5 times with a different chunk of the fold 
    serving as testing set. With each fold, three ANN are fitted, with 0, 1, 
    and 2 hidden layers. The in-sample and validation errors are returned.
    '''
    errIS = np.empty([3,len(chunks)]) # in-sample fit error
    errXV = np.empty([3,len(chunks)]) # cross vald error, [3 x nFolds]
    nIters = np.empty([3,len(chunks)], int) # n iters til convergence
    
    for ck in range(len(chunks)):
        # get index and dataset for current fold of cross-validation
        trnIdx = np.hstack([x for n,x in enumerate(chunks) if n != ck])
        vldIdx = np.hstack([x for n,x in enumerate(chunks) if n == ck])
        dataTrain,labelTrain = data[trnIdx],labels[trnIdx] # training
        dataTest,labelTest = data[vldIdx],labels[vldIdx] # validation
        
        if labels.ndim > 1: # if labels have more than one dimension
            labelTest = labelTest.argmax(axis=1) # collapse to 1D 

        # Fit and predict with ANN with 0 hidden layer
        wt0,nIters[0,ck],errIS[0,ck] = train_0hidd(dataTrain, labelTrain, eta,
                                                    shuffle=rand)
        errXV[0,ck] = errRate(pred_0hidd(dataTest, wt0), labelTest)

        # Fit and predict with ANN with 1 hidden layer
        wt1,nIters[1,ck],errIS[1,ck] = train_1hidd(dataTrain, labelTrain, eta, 
                                                    nHiddNd[0], shuffle=rand)
        errXV[1,ck] = errRate(pred_1hidd(dataTest,*wt1), labelTest)

        # Fit and predict with ANN with 2 hidden layers
        wt2,nIters[2,ck],errIS[2,ck] = train_2hidd(dataTrain, labelTrain, eta, 
                                                    nHiddNd[1], shuffle=rand)
        errXV[2,ck] = errRate(pred_2hidd(dataTest,*wt2), labelTest)

    return errIS,errXV,nIters