import sys, os
from functions import prepData, splitData
from crossValidation import crossValidate
from PredictPrune import PruneDTree
from Training import TrainDTree

path,fName = os.path.split(sys.argv[1])
if len(sys.argv) > 2: # to print the trees or not
	printTree = bool(sys.argv[2]) # command line argument
else:
	printTree = False # no print

if fName == 'car.data':
	carPath = os.path.join(path,fName)
	carNames = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety',
				'accept']
	data,features,classes = prepData(carPath, carNames, slice(-1),'accept')

elif fName == 'segmentation.data':
	segPD = os.path.join(path,fName)
	segVars = ['class', 'centCol', 'centRow', 'pixelCount', 'slDensity5', 
            'slDensity2', 'vedgeMean', 'vegdeSD', 'hedgeMean', 'hedgeSD',
            'intenseMean', 'rawredMean', 'rawblueMean', 'rawgreenMean',
             'exredMean', 'exblueMean', 'exgreenMean', 'valueMean', 
             'saturateMean', 'hueMean']

	segFeatIdx = [1,2] + list(range(4,19))
	data,features,classes = prepData(segPD, segVars, segFeatIdx, 'class')

elif fName == 'abalone.data':
	abalonePath = os.path.join(path,fName)
	abaloneNames = ['sex', 'length', 'diameter', 'height', 'wholeHt',
					'shuckWt', 'visceraWt', 'shellWt', 'rings']
	data,features,classNum = prepData(abalonePath, abaloneNames, 
									  slice(-1), 'rings')
	classes = classNum.astype(str)
	classes[classNum<=5] = '<5'
	classes[classNum>=16] = '16+'

else:
	sys.exit("No such data set.")


(xvData,xvLabel),xvFolds,pruningSet = splitData(data, classes)
fullErr,prnErr = crossValidate(xvData, xvLabel, pruningSet, xvFolds, printTree)

print("Full Tree - Mean Error: %f"%fullErr.mean())
print("Full Tree - St Dev Error: %f"%fullErr.std())

print("Pruned Tree - Mean Error: %f"%prnErr.mean())
print("Pruned Tree - St Dev Error: %f"%prnErr.std())


tr = TrainDTree(xvData, xvLabel) # train DTree using full cross-val sample
tr.combineChildNodes()  # combine subtrees with homogeneous classes
if printTree: # print full tree
	print("\n===full tree===")
	print(tr)
PruneDTree(tr, pruningSet[0], pruningSet[1]) # prune with pruning set
if printTree: # print pruned tree
	print("===pruned tree===")
	print(tr)