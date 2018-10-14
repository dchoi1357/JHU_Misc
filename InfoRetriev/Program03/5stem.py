import numpy as np
from scipy import stats
import pickle

def ismember(a_vec, b_vec):
    ''' MATLAB equivalent ismember function. Slower than above implementation'''
    b_dict = {b_vec[i]: i for i in range(0, len(b_vec))}
    indices = [b_dict.get(x) for x in a_vec if b_dict.get(x) is not None]
    booleans = np.in1d(a_vec, b_vec)
    return booleans, np.array(indices, dtype=int)


def loadDict(f):
	with open(f, 'rb') as fh:
		z = pickle.load(fh)
	return z
n = loadDict('sub_wu-a_scores.dict')
b = loadDict('sub_wu-b_scores.dict')


def pickResult(x, y, qID, N=100):
	topX,topY = [x[qID].most_common(N), y[qID].most_common(N)]
	xDocs,yDocs = ([x[0] for x in topX], [x[0] for x in topY])
	allDocs = np.union1d(xDocs, yDocs)
	
	xScores = [x if x else 0 for x in map(x[qID].get, allDocs)]
	yScores = [x if x else 0 for x in map(y[qID].get, allDocs)]
	
	return allDocs, xScores, yScores, len(xDocs)+len(yDocs)-len(allDocs)


def compMethod(N=100):
	print("Compareing results of Top %d Results"%N)
	nQry = len(n)
	results = np.zeros(nQry)
	for qID in range(nQry):
		x_y, x, y, inter = pickResult(n, b, qID, N)
		results[qID] = stats.wilcoxon(x,y)[1]
		print("%d,%.3f,%f"%(qID+1,inter/N,results[qID]))
	print('')
	
compMethod(10)
compMethod(20)
compMethod(50)
compMethod(100)

