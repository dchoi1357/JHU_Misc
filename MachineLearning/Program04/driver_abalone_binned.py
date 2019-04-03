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

if fName == 'abalone.data':
	abalonePath = os.path.join(path,fName)
	abaloneNames = ['sex', 'length', 'diameter', 'height', 'wholeHt',
					'shuckWt', 'visceraWt', 'shellWt', 'rings']
	data,features,classNum = prepData(abalonePath, abaloneNames, 
									  slice(-1), 'rings')
	classes = classNum.astype(str)
	classes[classNum<=8] = '1-8'
	classes[classNum>=11] = '11+'

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