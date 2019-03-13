import numpy as np
from knn import KNN
from functions import errRate

def tuneK(dataMat, labels, folds, categ, cnnSub=None):
	''' Tune for best K by finding K with the smallest classification error.
	The function loops through value of 1-10 and find the smallest K in that 
	given range which produces the smallest K. The function randomly takes
	one fold of the 5-fold crossvalidation as the testing set for this tuning.
	The K with the smallest associated error is returned.
	'''
	Ks = np.arange(1,11) # list of K valujes
	err = np.empty(len(Ks)) # pre-allocate erros
	
	pick = np.random.randint(len(folds)) # randomly pick fold as validation set
	trnIdx = np.hstack([x for n,x in enumerate(folds) if n != pick]) # train
	vldIdx = np.hstack([x for n,x in enumerate(folds) if n == pick]) # validate
	dataTrain,labelTrain = dataMat[trnIdx,:],labels[trnIdx] # training
	dataTest,labelTest = dataMat[vldIdx,:],labels[vldIdx] # validation

	if cnnSub is not None :
		dataTrain = dataMat[cnnSub]
		labelTrain = labels[cnnSub]
	
	for n,k in enumerate(Ks): # loop through K's
		pred = KNN(dataTrain, labelTrain, dataTest, k, categorical=categ)
		err[n] = errRate(pred, labelTest, categorical=categ)
	
	return Ks[np.argmin(err)], err # return K with smallest error

def getCrossValidFolds(dataMat, classVec, nFolds=5, categorical=False):
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

def crossValidate(dataMat, labels, chunks, k, categ, cnnSub=None):
	''' Perform n-fold cross validation.
	Given data matrix, labels, the chunks as folds, the function performs
	cross validation by using 4 out of 5 folds as training and the fifth as 
	testing set. This is repeated 5 times with a different chunk of the fold 
	serving as testing set. The error is returned.
	'''

	err = np.empty(len(chunks))
	for ck in range(len(chunks)):
		# get index and dataset for current fold of cross-validation
		trnIdx = np.hstack([x for n,x in enumerate(chunks) if n != ck])
		vldIdx = np.hstack([x for n,x in enumerate(chunks) if n == ck])
		dataTrain,labelTrain = dataMat[trnIdx,:],labels[trnIdx] # training
		dataTest,labelTest = dataMat[vldIdx,:],labels[vldIdx] # validation
		
		if cnnSub is not None: # using CNN
			dataTrain = dataMat[cnnSub]
			labelTrain = labels[cnnSub]

		pred = KNN(dataTrain, labelTrain, dataTest, k, categorical=categ)
		err[ck] = errRate(pred, labelTest, categorical=categ)
	return err