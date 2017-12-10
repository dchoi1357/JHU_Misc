import getTerms
import numpy as np

def parseTerms(s):
	return s[1:-1].split(', ')

#####################################

terms = getTerms.get() # get dictionary of terms for each subject

tmp = np.recfromcsv('filename_subject_list.csv') # read in list of file vs subj
docSubj = dict(zip(tmp['filename'], tmp['subject'])) # dict of name and subj

tmp = np.recfromtxt('./outFiles/part-r-00000', delimiter='\t') # output and topN
files = tmp[:,0]
topN = tmp[:,1:]

val = np.empty(topN.shape, dtype=bool)
for n,(f,t) in enumerate(zip(files,topN)): # see if topN term within subject
	val[n] = np.in1d(t, terms[ docSubj[f] ])

head = np.char.add('term', np.char.mod('%d', range(1,val.shape[1]+1)) ) # header
valString = np.char.mod('%i', val) # convert True/False into '1'/'0' 

np.savetxt('./outFiles/validation.csv', np.hstack((files[:, None], valString)),
	fmt='%s', delimiter=',', header='fileName,'+','.join(head), comments='')
