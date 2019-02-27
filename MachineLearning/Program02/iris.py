import os
import numpy as np, pandas as pd
from stepForwSelect import ForwardSelect
from functions import prepData

irisFile = os.path.join('./data/', 'iris.data')
irisName = ['sepalLen', 'sepalWth', 'petalLen', 'petalWth', 'class']
raw = pd.read_csv(irisFile , names=irisName)  # read CSV file

irisFeats = irisName[:-1] # list of feature names
irisMat = raw[irisFeats].values # 2d-array of feature values
irisK = len(raw['class'].unique()) # number of classes

ForwardSelect(irisMat, irisK) # run algorithm