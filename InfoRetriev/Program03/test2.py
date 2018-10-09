import sys
import pickle, struct, math, re
from collections import Counter
import tokenHelper as tkn

queryFileName = './data/cds14.topics.txt'
invFileName = './data/cds14_nstem.inv'
dictFileName = './data/cds14_nstem.dict'
tokenizeAlgoNum = '3'
tg = re.compile('<q id=\\d+>') # regex to match leading tag
vcb = dict() # corpus to be added as queries are looked up

def getQueries(fname):
	with open (fname, 'r') as f:
		rawStr = f.read().casefold() # to low case and remove LF
		rawStr = tg.sub('', rawStr) # remove lead tag
		rawStr = rawStr.split('</q>') # split by end tag
	return [x for x in rawStr if x.strip()] # return non-empty paragraphs

def parseQuery(qtxt):
	return Counter( tkn.tokenize(qtxt,tokenizeAlgoNum) )

def readBinFileIntoInts(fName, offset, nInts):
	with open(fName, 'rb') as f:
		f.seek(offset*4)
		tmp = [x[0] for x in struct.iter_unpack('<I', f.read(4*2*nInts))]
	return list(zip(tmp[0::2], tmp[1::2]))

def lookupPostingIDF(term):
	df,offset,idf = d[term]
	posting = readBinFileIntoInts(invFileName, offset, df)
	return ( posting, idf )


def cosineSimScore(qDct):
	sims = Counter() # counter for storing simularity scores
	for tk in qDct: 
		if tk not in vcb: # if have not been looked up before
			vcb[tk] = lookupPostingIDF(tk) # lookup posting from inv file
		postings,idf = vcb[tk] # get posting list and IDF for term
		for docID,tf in postings: # iterate through posting list
			sims[docID] += tf*idf * qDct[tk]*idf / docLen[docID]
	return sims


with open(dictFileName, 'rb') as h:
	d = pickle.load(h)
docLen = d['#docLen#']
qDict = parseQuery( getQueries(queryFileName)[0] )
z = cosineSimScore(qDict)
o = [(tk,vcb[tk][1]*qDict[tk]) for tk in qDict]

