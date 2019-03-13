import numpy as np, pandas as pd

def pairwiseDist(x, y=None):
	''' Calculate all pair-wise distance between input vectors.
	If only one argument, the distance of every vector with every other vector
	within the same matrix is calculated. If two arguments, the pairs are the
	cartesian product of all vectors in the two arguments
	'''	
	if y is None: # if only one input, take itself as second input
		y = x
	return np.sum((x[:,None]-y)**2,axis=2)**0.5 # square root of squared diffs

def prepData(dataPathDir, fieldNames, featSlices, labelName, 
			 sep=',', transf=None):
	raw = pd.read_csv(dataPathDir , sep=sep, names=fieldNames) # read dlm file
	if isinstance(featSlices, slice):
		dataFeats = fieldNames[featSlices] # list of feature names
	else:
		dataFeats = [fieldNames[i] for i in featSlices]
	if transf is None: # no transformation
		dataMat = raw[dataFeats].values # original values
	elif transf.lower() == 'std' : # if choose to standardize data
		meanVals = raw[dataFeats].mean().values # mean of all features
		stdVals = raw[dataFeats].std().values # standard deviations
		dataMat = (raw[dataFeats].values - meanVals) / stdVals # [x-E(x)]/S(X)
	elif transf.lower() == 'rescale': # rescale to values in [0,1]
		mins = raw[dataFeats].min().values # min of feature vals
		maxs = raw[dataFeats].max().values # max of feature vals
		dataMat = (raw[dataFeats].values-mins) / (maxs-mins) # x-min/range(x)
	else: # error out
		raise Exception('No such transformation available')
	return dataMat,dataFeats,raw[labelName].values

def errRate(pred, actual, categorical=True):
	if categorical: # if categ., return classification err rate
		return sum(pred!=actual) / pred.size
	else: # if numeric, return RMSE
		return np.linalg.norm(pred-actual)/np.sqrt(pred.size)