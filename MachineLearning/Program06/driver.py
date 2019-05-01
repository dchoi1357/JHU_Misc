import sys, os
import pandas as pd, numpy as np
from utilities import discretizeMean, oneHot, normalizeDF, makeClassMat
from crossValidate import getXVFolds, doCrossValidate, tuneNHiddNodes

path,fName = os.path.split(sys.argv[1])

stochastic = False
if len(sys.argv) > 2:
    stochastic = int(sys.argv[2].strip()) > 0


if fName == 'glass.txt':
    glassData = os.path.join(path, fName)
    glassNames = ['id','RI','Na','Mg','Al','Si','K','Ca','Ba','Fe','type']
    raw = pd.read_csv(glassData , names=glassNames ) # read in glass file
    # create multidimensional class matrix
    classVec = pd.concat([raw['type'].isin([1,3]), raw['type'].isin([2,4]), 
                          raw['type'].isin([5,6,7])], axis=1).values * 1
    feats = glassNames[1:-1]
    dataMat = normalizeDF(raw[feats])

elif fName == 'iris.data':
    irisFile = os.path.join(path, fName)
    irisName = ['sepalLen', 'sepalWth', 'petalLen', 'petalWth', 'class']
    raw = pd.read_csv(irisFile , names=irisName)  # read CSV file
    irisTypeVec = raw['class']
    classVec = makeClassMat(raw['class'])
    feats = irisName[:-1]
    dataMat = normalizeDF(raw[feats])

elif fName == 'breast-cancer-wisconsin.data':
    bc_WI_data = os.path.join(path, fName)
    bc_WI_names = ['id', 'clumpThick', 'unifSize', 'unifShape', 'margAdhsn', 
                   'epithSize', 'bareNuclei', 'blandChrom', 'normNucleo', 
                   'mitoses', 'class']
    raw = pd.read_csv(bc_WI_data , names=bc_WI_names)  # read CSV file
    missRow = (raw=='?').any(axis=1).values # rows with missing data
    raw = raw[~missRow] # remove rows with missing
    raw = raw.apply(pd.to_numeric, errors= 'coerce') # conv to numeric data
    bcFeats = bc_WI_names[1:-1] # list of feature variables
    classVec = makeClassMat(raw['class']==4)
    dataMat = normalizeDF(raw[bcFeats])
    

elif fName == 'house-votes-84.data':
    vote84Data = os.path.join(path, fName)
    vote84Names = ['party', 'infant', 'water', 'budget', 'doctorfee', 'salvador',
                    'religion', 'satellite', 'contras', 'missile', 'immigration',
                    'synfuels', 'education', 'superfund', 'crime', 'exports',
                    'ZAF']
    raw = pd.read_csv(vote84Data , names=vote84Names ) # read in vote file
    oneHotCols = oneHot(raw,['water','education','ZAF'])
    # remove variables with completed one-hot coding from list of variables
    yesVars = np.setdiff1d(vote84Names[1:],['water','education','ZAF'])
    yesVote = raw.loc[:,yesVars] == 'y' # boolean for vote='yes' for rest of vars
    yesVote.columns = [s+'_y' for s in yesVote.columns]
    voteData = pd.concat([yesVote,oneHotCols], axis=1) # concat two dataframes
    dataMat = voteData.values * 1 # give matrixs of 0 & 1 for calculation

    repub = raw['party']=='republican' # boolean for republicans
    classVec = makeClassMat(repub)# vector of 0 & 1 for calculation

elif fName == 'soybean-small.data':
    soyData = os.path.join(path, fName)
    # use cardinal number for feature names, like c01 for 1st col, etc
    soyNames = ['c%02d'%(n+1) for n in range(35)] + ['class'] 
    raw = pd.read_csv(soyData, names=soyNames)
    feats = np.array(soyNames)[raw.nunique()!=1] # remove feats with only 1 value
    feats = feats[raw[feats].nunique() == 2] # remove if non-binomial features

    tmpDF = pd.DataFrame()
    for f in feats: # loop over features
        tmpDF[f] = (raw[f] == raw[f].unique()[0]) # if feature is first uniq val
    dataMat = tmpDF.values * 1 # change to 1's and 0's
    classVec = makeClassMat(raw['class']) # all classes
else:
    sys.exit("No such data set.")

folds = getXVFolds(dataMat, classVec, categorical=True) # stratified sampling

nHiddNodes = tuneNHiddNodes(dataMat, classVec, folds, stochastic)
print("N Hidden Nodes: %s"%nHiddNodes)

errInSamp,errXV,nIters = doCrossValidate(dataMat, classVec, folds, nHiddNodes,
                                         stochastic)

print("In Sample Err - Mean Error: %s"%errInSamp.mean(axis=1))
print("In Sample Err - Std Dev: %s"%errInSamp.std(axis=1))

print("Cross Vald Err - Mean Error: %s"%errXV.mean(axis=1))
print("Cross Vald Err - Std Dev: %s"%errXV.std(axis=1))

print("Mean N Iters to Convergence: %s"%nIters.mean(axis=1))
