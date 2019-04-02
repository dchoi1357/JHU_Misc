import numpy as np
from Training import TrainDTree
from PredictPrune import PredictDTree, PruneDTree

def errRate(pred, actual, categorical=True):
    if categorical: # if categ., return classification err rate
        return sum(pred!=actual) / pred.size
    else: # if numeric, return RMSE
        return np.linalg.norm(pred-actual)/np.sqrt(pred.size)

def getXVFolds(dataMat, classVec, nFolds=5, categorical=False):
	''' Cut N-fold cross validation of the data set
	Given a data matrix, a class vector, and the number of folds, the function
	randomly cuts a 5-fold cross validation. If the data is categorical, 
	stratified sampling is used.
	'''
	
	idx = np.arange(dataMat.shape[0]) # construct index of data rows
	if categorical:
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

def crossValidate(data, labels, pruneSet, chunks, categ=True):
    ''' Perform n-fold cross validation.
    Given data matrix, labels, the chunks as folds, the function performs
    cross validation by using 4 out of 5 folds as training and the fifth as 
    testing set. This is repeated 5 times with a different chunk of the fold 
    serving as testing set. The error is returned.
    '''
	
    prnData,prnLabels = pruneSet

    errFull = np.empty(len(chunks))
    errPrune = np.empty(len(chunks))
    for ck in range(len(chunks)):
        # get index and dataset for current fold of cross-validation
        trnIdx = np.hstack([x for n,x in enumerate(chunks) if n != ck])
        vldIdx = np.hstack([x for n,x in enumerate(chunks) if n == ck])
        dataTrain,labelTrain = data[trnIdx],labels[trnIdx] # training
        dataTest,labelTest = data[vldIdx],labels[vldIdx] # validation
        
        tr = TrainDTree(dataTrain, labelTrain) # train DTree
        tr.combineChildNodes() # combine subtrees with homogeneous classes
        pred1 = PredictDTree(tr, dataTest) # predict with full tree
        
        PruneDTree(tr, pruneSet[0], pruneSet[1]) # prune with pruning set
        pred2 = PredictDTree(tr, dataTest) # predict with pruned tree
        
        errFull[ck] = errRate(pred1, labelTest) # error with full tree
        errPrune[ck] = errRate(pred2, labelTest) # error with pruned tree
    return errFull,errPrune