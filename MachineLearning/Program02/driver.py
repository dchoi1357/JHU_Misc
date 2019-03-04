import sys, os
from functions import prepData, printResults
from stepForwSelect import ForwardSelect
from bruteForce import BruteForceSelect

path,fName = os.path.split(sys.argv[1])

if fName == 'iris.data':
	irisData = os.path.join(path,fName)
	irisName = ['sepalLen', 'sepalWth', 'petalLen', 'petalWth', 'class']
	dataMat,feats,nK,dataMeans = prepData(irisData,irisName,slice(0,-1))
elif fName == 'glass.data':
	glassData = os.path.join(path,fName)
	glassNames = ['id','RI','Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe',
				'class']
	dataMat,feats,nK,dataMeans = prepData(glassData,glassNames,slice(1,-1))
elif fName == 'spambase.data':
	spamData = os.path.join(path,fName)
	spamNames = ['make', 'address', 'all', '3d', 'our', 'over', 'remove',
		'internet', 'order', 'mail', 'receive', 'will', 'people', 'report',
		'addresses', 'free', 'business', 'email', 'you', 'credit', 'your', 
		'font', '0', 'money', 'hp', 'hpl', 'george', '650', 'lab', 'labs', 
		'telnet', '857', 'data', '415', '85', 'technology', '1999', 'parts', 
		'pm', 'direct', 'cs', 'meeting', 'original', 'project', 're', 'edu', 
		'table', 'conference', 'semicolon', 'paren', 'bracket', 'exclaim', 
		'dollar', 'pound', 'capsAvg', 'capsMax', 'capsTotal', 'class']
	dataMat,feats,nK,dataMeans = prepData(spamData,spamNames,slice(-4))
else:
	pass

fsOut = ForwardSelect(dataMat, nK, trace=True)
print('\nGroups: %s\n'%fsOut[2])
printResults(fsOut, feats)
print("\nBrute Force Results:")
bruteOut = BruteForceSelect(dataMat, nK)
if bruteOut is None:
	print("\tData too large for brute forcing.")
else:
	print("\tBest coeff: %f"%bruteOut[-1])
	printResults(bruteOut, feats)