import getTerms
terms = getTerms.get()

tmp = np.recfromcsv('filename_subject_list.csv')
docSubj = dict(zip(tmp['filename'], tmp['subject']))

tmp = np.recfromtxt('./outFiles/part-r-00000', delimiter='\t')
files = tmp[:,0]
topN = np.array([parseTerms(x) for x in tmp[:,1]])

def parseTerms(s):
	return s[1:-1].split(', ')

val = np.empty(topN.shape, dtype=bool)
for n,(f,t) in enumerate(zip(files,top3)):
	val[n] = np.in1d(t, terms[ docSubj[f] ])

head = np.char.add('term', np.char.mod('%d', range(1,val.shape[1]+1)) ) # header
valString = np.char.mod('%i', val) # convert True/False into '1'/'0' 

np.savetxt('validation.csv', np.hstack((files[:, None], valString)), fmt='%s',
	delimiter=',', header='fileName,'+','.join(head), comments='')
