import sys, os
from functions import prepData
from crossValidate import crossValidate, tuneK, getCrossValidFolds
from knn import KNN, consistentSubset
import numpy as np

path,fName = os.path.split(sys.argv[1])
toTrans = int(sys.argv[2]) > 0
if toTrans:
	func = 'rescale'
else:
	func = None

cnn = False
if fName == 'ecoli.data':
	ecoliPD = os.path.join(path,fName)
	ecoliVars = ['seq', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 
				 'class']
	ecoliMat,ecoliFeat,ecoliY = prepData(ecoliPD, ecoliVars, slice(1,-1), 
										"class", r'\s+', transf=func)
	folds = getCrossValidFolds(ecoliMat, ecoliY, categorical=True)
	minK,errs = tuneK(ecoliMat, ecoliY, folds, categ=True)
	xvErrs = crossValidate(ecoliMat, ecoliY, folds, minK, categ=True)

	subset,dataIdx = consistentSubset(ecoliMat, ecoliY) # find cons. subset
	minCNNK,errs = tuneK(ecoliMat, ecoliY, folds, True, subset)
	xvCErrs = crossValidate(ecoliMat, ecoliY, folds, minCNNK, True, subset)
	cnn = True

elif fName == 'forestfires.data':
	firePD = os.path.join(path,fName)
	fireVars = ['X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp',
			'RH', 'wind', 'rain', 'area']
	fireIdx = [0,1] + list(range(4,12))
	fireMat,fireFeats,fireY = prepData(firePD, fireVars, fireIdx, "area", transf=func)
	fireY = np.log(fireY+1)
	folds = getCrossValidFolds(fireMat, fireY, categorical=False)
	minK,errs = tuneK(fireMat, fireY, folds, categ=False)
	xvErrs = np.exp(crossValidate(fireMat, fireY, folds, minK, categ=False))-1

elif fName == 'machine.data':
	machPD = os.path.join(path,fName)
	machVars = ['vendor', 'model', 'MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 
				'CHMAX', 'PRP', 'ERP']
	machMat,machFeat,machY = prepData(machPD, machVars, slice(2,-2), "PRP", 
					transf=func)
	folds = getCrossValidFolds(machMat, machY, categorical=False)
	minK,errs = tuneK(machMat, machY, folds, categ=False)
	xvErrs = crossValidate(machMat, machY, folds, minK, categ=False)

elif fName == 'segmentation.data':
	segPD = os.path.join(path,fName)
	segVars = ['class', 'cent-col', 'cent-row', 'pixel-count', 'sl-density-5', 
				'sl-density-2', 'vedge-mean', 'vegde-sd', 'hedge-mean', 'hedge-sd',
				'intense-m', 'rawred-m', 'rawblue-m', 'rawgreen-m', 'exred-m', 
				'exblue-m', 'exgreen-m', 'value-m', 'saturate-m', 'hue-m']
	segIdx = [1,2] + list(range(4,19))
	segMat,segFeat,segY = prepData(segPD, segVars, segIdx, "class", transf=func)
	folds = getCrossValidFolds(segMat, segY, categorical=True)
	minK,errs = tuneK(segMat, segY, folds, categ=True)
	xvErrs = crossValidate(segMat, segY, folds, minK, categ=True)
	
	subset,dataIdx = consistentSubset(segMat, segY) # find cons. subset
	minCNNK,errs = tuneK(segMat, segY, folds, True, subset)
	xvCErrs = crossValidate(segMat, segY, folds, minCNNK, True, subset)
	cnn = True
else:
	pass

print("Result of KNN, Rescale=%r:"%toTrans)
print("Best K=%u"%minK)
print("Average Classification Errors: %f"%xvErrs.mean())
print("St Dev of Classification Errors: %f"%xvErrs.std())

if cnn:
	print("Best K in CNN=%u"%minCNNK)
	print("Average CNN Classification Errors: %f"%xvCErrs.mean())
	print("St Dev of CNN Classification Errors: %f"%xvCErrs.std())