import textwrap, sys, random, getTerms
import numpy as np

t = getTerms.get() # get terms from getTerms.py
subjects = t.keys() # subjects
terms = [t[s] for s in t] # list of list of terms
punctuations = [' ', ', ', '. ']

wordCt = [100, 300]

words = np.recfromcsv('common_words_freq.csv') # read CSV
p = np.true_divide(words['frequency'], np.sum(words['frequency']))

probs1 = np.array([0.8, 0.05, 0.15]) # texts, terms, specific terms
probs2 = np.array([0.8, 0.1, 0.1]) # prob of space vs comma vs period

def genDoc(spec): # spec argument is a list of specialized terms
	ct = np.random.randint(wordCt[0], wordCt[1]) # word count for doc
	strc = np.random.choice(range(3), ct, True, probs1) # 0=wrd, 1=trm, 2=spec

	txt = np.empty(ct, dtype=words.dtype[0])
	txt[strc==0] = np.random.choice(words['word'], np.sum(strc==0), True, p)
	txt[strc==1] = np.random.choice(np.hstack(terms), np.sum(strc==1), True)
	txt[strc==2] = np.random.choice(spec, np.sum(strc==2), True)
	
	puncts = np.random.choice(punctuations, ct, True, probs2)
	
	out = np.empty(txt.size*2, dtype=txt.dtype)
	out[0::2] = txt
	out[1::2] = puncts
	out[-1] = '.' # set last punctuation to period.

	return textwrap.wrap(''.join(out), 80)


def printListOfTxt(l,name):
	with open('./docs/'+name, 'w') as f:
		f.writelines('\n'.join(l))


def genRandName(length, n=1):
	fmt = '%' + '0%dx' % length
	return [fmt % random.randrange(16**length) for x in range(n)]


if len(sys.argv) > 0:
	nDocs = int(sys.argv[1])
else:
	nDocs = 20

fileNames = np.char.add(genRandName(10, nDocs), '.txt')
subjIdx = np.random.randint(len(subjects), size=nDocs)
name_x_subj = np.vstack( (fileNames,np.array(subjects)[subjIdx]) ).T

for n,idx in enumerate(subjIdx):
	printListOfTxt(genDoc(terms[idx]), fileNames[n])

np.savetxt('filename_subject_list.csv', name_x_subj, fmt='%s', delimiter=',',
	header='fileName,subject', comments='')

