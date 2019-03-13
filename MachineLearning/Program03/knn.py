import numpy as np
from functions import pairwiseDist

def mostCommonElem(elems):
	''' Get the most common element given a list of elements
	Using a dict, keep track of the counts of various elements. Return the most
	frequent element. If there's a tie, then return a random one.
	'''
	counts = dict() # dict to keep track of counts
	for e in elems: # loop over array
		counts[e] = counts.get(e,0) + 1 # increase count by 1 (def count of 0)
	maxCount = -1
	for e,ct in counts.items(): # loop over counts, set maxCount if count larger
		maxCount = ct if ct > maxCount else maxCount
	# get list of elems which has same count as maxCount (if multiple elems)
	mostFreqElem = [e for e,ct in counts.items() if ct==maxCount]
	return np.random.choice(mostFreqElem) # randomly choose one elemn from all

def kMinValIdx(mat, k):
	''' Given a matrix, return the k smallest index and elements of each row.
	This is accomplished by performing k bubbles of the bubble sort. In each
	bubble, an element is exchanged with its right if its smaller, and the 
	algorithm goes down the list of element until the end. As the bubbling is
	performed k times, it is guaranteed that the k left-most elements will be
	in sorted order.
	'''
	mat = np.copy(mat) # create copy of variable
	if mat.size == 1: # if only one elem, that element is the max
		return np.array([[0]]),mat
	if mat.ndim == 1: # reshape if one dimensional vector
		mat = mat.reshape([1,-1])
	idx = np.ones(mat.shape,int).cumsum(axis=1)-1 # rows of idx: 0,1,...,nCol
	
	for it in range(k): # perform k bubbles to get k smallest
		for col in range(mat.shape[1]-it-1):
			toSwap = mat[:,col] < mat[:,col+1] # if elem smaller than next elem
			# swap cols of data matrix and matrix of indices
			mat[np.ix_(toSwap,[col,col+1])] = mat[np.ix_(toSwap,[col+1,col])]
			idx[np.ix_(toSwap,[col,col+1])] = idx[np.ix_(toSwap,[col+1,col])]
	return idx[:,-k:],mat[:,-k:] # return smallest elemenst per row and the idxs

def KNN(trainX, trainY, testX, K, categorical):
	''' K-nearest Neighbors classifier.
	Arguments are: the training data, training label, test data, the K 
	hyperparameter, and a boolean variable indicating whether the data is 
	categorical. The function first calculate all the pair-wise distance 
	between the test data points and the training data point. It then finds
	the closest K data points in the training data set. The lables of these 
	K data points are then either: 1) taken an average of if the data is not
	categorical, or 2) taken the plurality of if it's categorical
	'''
	dists = pairwiseDist(testX, trainX) # all pairwise dist of two datasets
	knnIdx,_ = kMinValIdx(dists, K) # idx of K closest data pts in training set
	knnLabels = trainY[knnIdx] # labels of these closest data points
	
	testY = np.empty(testX.shape, trainY.dtype) # pre-allocate test data labels
	if not categorical: # regression, calculate mean
		testY = knnLabels.mean(axis=1) # mean of k-closest label values
	else: # classification, get most common class label
		testY = np.array([mostCommonElem(lab) for lab in knnLabels])
	return testY # return results

#########
# The following code relate to condensed-nearest neighbors

def pickAndRemove(arr):
	''' Given an array, remove one random element. Return the removedelement
	as well	as the unpicked elements in a separate list.
	'''
	j = np.random.randint(arr.size)
	return arr[j], np.delete(arr,j)
	
def consistentSubset(trainX, trainY, K=1):
	''' Using Hart's algorithm to find a consistent subset.
	Arguments are: training data, training label, and K. K is default to one
	as per the original Hart's algorithm. The algorithm randomly picks 
	'''

	dists = pairwiseDist(trainX) # all pairwise dist of two datasets
	idx = np.arange(trainX.shape[0]) # construct index of data rows
	Z,idx = pickAndRemove(idx) # randomly pick 1st pt of of subset 

	converged = False
	while not converged:
		converged = True # stop unless a misclassification
		np.random.shuffle(idx) # shuffle sequence of sample to train randomly
		
		for x in idx: # loop over all samples
			nnIdx = kMinValIdx(dists[x,Z], 1)[0] # idx of NN in Z
			nnLabel = trainY[nnIdx].flatten() # label of NN of x in Z
			if nnLabel!=trainY[x]: # if misclassification
				Z = np.hstack([Z,x]) # add to consistent subset
				converged = False # continue training
		idx = np.setdiff1d(idx, Z) # remove training set from samples
		
	return Z, idx
