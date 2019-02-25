import os
import numpy as np, pandas as pd
from stepForwSelect import ForwardSelect

spamData = os.path.join('./data/', 'spambase.data')
spamNames = ['make', 'address', 'all', '3d', 'our', 'over', 'remove',
	'internet', 'order', 'mail', 'receive', 'will', 'people', 'report',
	'addresses', 'free', 'business', 'email', 'you', 'credit', 'your', 'font',
	'0', 'money', 'hp', 'hpl', 'george', '650', 'lab', 'labs', 'telnet', '857',
	'data', '415', '85', 'technology', '1999', 'parts', 'pm', 'direct', 'cs',
	'meeting', 'original', 'project', 're', 'edu', 'table', 'conference',
	'semicolon', 'paren', 'bracket', 'exclaim', 'dollar', 'pound', 'capsAvg',
	'capsMax', 'capsTotal', 'class']
raw = pd.read_csv(spamData , names=spamNames)  # read CSV file

spamFeats = spamNames[:-1] # list of feature names
spamMat = raw[spamFeats].values # 2d-array of feature values
spamK = len(raw['class'].unique()) # number of classes

ForwardSelect(spamMat, spamK) # run algorithm