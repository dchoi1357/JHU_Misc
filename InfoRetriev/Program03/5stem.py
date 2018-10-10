import numpy as np
from scipy import stats

aFile = 'wu-a.txt'
bFile = 'wu-b.txt'

n = np.genfromtxt(aFile, delimiter=' ', dtype=None)
b = np.genfromtxt(bFile, delimiter=' ', dtype=None)

def ismember(a_vec, b_vec):
    ''' MATLAB equivalent ismember function. Slower than above implementation'''
    b_dict = {b_vec[i]: i for i in range(0, len(b_vec))}
    indices = [b_dict.get(x) for x in a_vec if b_dict.get(x) is not None]
    booleans = np.in1d(a_vec, b_vec)
    return booleans, np.array(indices, dtype=int)


def pickResult(x, y, qID):
	docs1 = x['f2'][x['f0'] == qID]
	ranks1 = x['f3'][x['f0'] == qID]
	docs2 = y['f2'][y['f0'] == qID]
	ranks2 = y['f3'][y['f0'] == qID]
	allDocs = np.union1d(docs1, docs2)

	in1_all,in1_idx = ismember(allDocs, docs1)
	in2_all,in2_idx = ismember(allDocs, docs2)

	all_rank1 = np.zeros(len(allDocs))
	all_rank2 = np.zeros(len(allDocs))

	all_rank1[in1_all] = ranks1[in1_idx]
	all_rank1[~in1_all] = 100+(len(allDocs) - len(docs1))/2

	all_rank2[in2_all] = ranks2[in2_idx]
	all_rank2[~in2_all] = 100+(len(allDocs) - len(docs2))/2
	
	return allDocs, all_rank1, all_rank2, len(allDocs)-len(docs1)

nQry = max(n['f0'])
results = np.zeros(nQry)
for qID in range(nQry):
	x_y, x, y, d = pickResult(n, b, qID+1)
	results[qID] = stats.wilcoxon(x,y)[1]
	print("%d,%f"%(d,stats.wilcoxon(x,y)[1]))
	


