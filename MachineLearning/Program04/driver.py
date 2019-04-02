import sys, os
from functions import prepData, splitData
from crossValidation import crossValidate

path,fName = os.path.split(sys.argv[1])

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

elif fName == 'machine.data':
	abalonePath = os.path.join(path,fName)
	abaloneNames = ['sex', 'length', 'diameter', 'height', 'wholeHt',
					'shuckWt', 'visceraWt', 'shellWt', 'rings']
	data,features,classes = prepData(abalonePath, abaloneNames, 
									 slice(-1),'rings')
	classes = classes.astype(str)
	classes[ringVec<=5] = '<5'
	classes[ringVec>=16] = '16+'

else:
	pass


(xvData,xvLabel),xvFolds,pruningSet = splitData(carData, acceptVec)
fullErr,prnErr = crossValidate(xvData, xvLabel, pruningSet, xvFolds)

print("Full Tree - Mean Error: %f"%fullErr.mean())
print("Full Tree - St Dev Error: %f"%fullErr.std())

print("Pruned Tree - Mean Error: %f"%prnErr.mean())
print("Pruned Tree - St Dev Error: %f"%prnErr.std())
