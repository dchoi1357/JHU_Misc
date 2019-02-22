import numpy as np
import os
import pandas as pd
import functions.modeling as mdl # import modeling functions

irisFile = os.path.join('./data/', 'iris.data')
irisName = ['sepalLen', 'sepalWth', 'petalLen', 'petalWth', 'class']
raw = pd.read_csv(irisFile , names=irisName)  # read CSV file

irisFeats = irisName[:-1]
meanVals = raw[irisFeats].mean() # mean value for every feature
irisData = pd.DataFrame() # pre-allocate data frame for data
for v in irisFeats: # create DF of features if they are > mean feat value
    irisData[v] = raw[v] > meanVals[v]
irisMat = irisData.values * 1 # feature vector of 0 & 1s
setosa = raw.loc[:,["class"]]=='Iris-setosa' # Iris-setosa class
setosaVec = setosa.values.ravel()*1 # class vectors of 0 & 1s

irisPrm = {'theta': irisMat.shape[1]/4, 'alpha': 2} # hyper-parameters
mdl.crossValidate(irisMat, setosaVec, irisPrm)
wts,nbProb = mdl.estModels(irisMat,setosaVec,irisPrm)
print()
mdl.printWinnowModel(wts, irisData.columns)
mdl.printNBmodel(nbProb, irisData.columns)
