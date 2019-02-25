import os
import numpy as np, pandas as pd
from stepForwSelect import ForwardSelect

glassData = os.path.join('./data/', 'glass.data')
glassNames = ['id','RI','Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'class']
raw = pd.read_csv(glassData , names=glassNames)  # read CSV file

glassFeats = glassNames[1:-1] # list of feature names
glassMat = raw[glassFeats].values # 2d-array of feature values
glassK = len(raw['class'].unique()) # number of classes

ForwardSelect(glassMat, glassK) # run algorithm