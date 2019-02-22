import numpy as np
import os
import pandas as pd
import functions.modeling as mdl # import modeling functions

bc_WI_data = os.path.join('./data/', 'breast-cancer-wisconsin.data')
bc_WI_names = ['id', 'clumpThick', 'unifSize', 'unifShape', 'margAdhsn', 
               'epithSize', 'bareNuclei', 'blandChrom', 'normNucleo', 
               'mitoses', 'class']
raw = pd.read_csv(bc_WI_data , names=bc_WI_names)  # read CSV file
raw = raw.apply(pd.to_numeric, errors= 'coerce') # convert all to numeric
bcFeats = bc_WI_names[1:-1] # list of feature variables

meanVals = raw[bcFeats].mean() # mean value for every feature
bcData = pd.DataFrame() # pre-allocate data frame for data
for v in bcFeats: # create DF of features if they are > mean feat value
    bcData[v] = raw[v] > meanVals[v]
bcMat = bcData.values * 1 # feature vector of 0 & 1s
malign = raw.loc[:,["class"]]==4 # if case is malignant tumor
malignVec = malign.values.ravel()*1 # class vectors of 0 & 1s

brCanPrm = {'theta': bcMat.shape[1]/4, 'alpha': 2} # hyper-parameters
mdl.crossValidate(bcMat, malignVec, brCanPrm)
wts,nbProb = mdl.estModels(bcMat,malignVec,brCanPrm) # estimate with all data
print()
mdl.printWinnowModel(wts, bcData.columns)
mdl.printNBmodel(nbProb, bcData.columns)