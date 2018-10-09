import sys
if len(sys.argv) != 6: # check number of commandline inputs
	print(("Usage: %s [queryFileName] [invFileName] [dictName] " + \
		"[tokenizeAlgoNumber] [outFileName]") %  sys.argv[0])
	sys.exit()

import pickle, struct, math, re
from collections import Counter
import tokenHelper as tkn

tg = re.compile('<q id=\\d+>') # regex to match leading tag
vcb = dict() # corpus to be added as queries are looked up
queryFileName,invFileName,dictFileName,tokenizeAlgoNum,outFileName = \
	[ sys.argv[n] for n in range(1,len(sys.argv)) ]

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
	
def getTopNSimDocs(qID, simScore, N=100):
	topN = simScore.most_common(N)
	fmt = '%d Q0 %d %d %f Wu\n'
	return [fmt % (qID,docID,n+1,score) for n,(docID,score) in enumerate(topN)]


with open(dictFileName, 'rb') as h:
	d = pickle.load(h)
docLen = d['#docLen#']
print("Parsing query file: %s"%queryFileName)
print("\tQuerying inverted file %s and dict %s"%(invFileName,dictFileName))
with open(outFileName, 'w') as fh:
	for qInd,qTxt in enumerate(getQueries( queryFileName )):
		qDict = parseQuery(qTxt)
		simScores = cosineSimScore(qDict)
		out = getTopNSimDocs(qInd+1, simScores)
		fh.writelines(out)
print("Wrote out query results to: %s" % outFileName)

# for x in lookups:
# 	lookupAndPrint(x[0], x[1])
