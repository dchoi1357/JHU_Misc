import sys
if len(sys.argv) != 6: # check number of commandline inputs
	print(("Usage: %s [queryFileName] [invFileName] [dictName] " + \
		"[tokenizeAlgoNumber] [outFileName]") %  sys.argv[0])
	sys.exit()

import pickle, struct, math, re
from collections import Counter
import tokenHelper as tkn # import custom function for tokenization of text

# command line arguments
queryFileName,invFileName,dictFileName,tokenizeAlgoNum,outFileName = \
	[ sys.argv[n] for n in range(1,len(sys.argv)) ]
tg = re.compile('<q id=\\d+>') # regex to match leading tag
vcb = dict() # corpus to be added as queries are looked up

def getQueries(fname): # get list of query texts from entire file
	with open (fname, 'r') as f:
		rawStr = f.read().casefold() # to low case and remove LF
		rawStr = tg.sub('', rawStr) # remove lead tag
		rawStr = rawStr.split('</q>') # split by end tag
	return [x for x in rawStr if x.strip()] # return non-empty paragraphs

def parseQuery(qtxt): # parse (tokenize) one individual querys
	return Counter( tkn.tokenize(qtxt,tokenizeAlgoNum) )

def readBinFileIntoInts(fName, offset, nInts): # read binary inverted file
	with open(fName, 'rb') as f:
		f.seek(offset*4) # seek to the correct offset, and read multiple 4-bytes
		tmp = [x[0] for x in struct.iter_unpack('<I', f.read(4*2*nInts))]
	return list(zip(tmp[0::2], tmp[1::2])) # return as posting list

def lookupPostingIDF(term): # lookup posting list and IDF for single term
	df,offset,idf = d[term] # find DF and offset from vocab dict
	posting = readBinFileIntoInts(invFileName, offset, df) # read inverted file
	return ( posting, idf ) # return posting list and IDF for term

# calculcate cosine score using a query counter dict, containing number of 
# times a term appear in a query
def cosineSimScore(qDct): 
	sims = Counter() # counter for storing simularity scores
	for tk in qDct: # loop over terms in a query
		if tk not in vcb: # if have not been looked up before
			vcb[tk] = lookupPostingIDF(tk) # lookup posting from inv file
		postings,idf = vcb[tk] # get posting list and IDF for term
		for docID,tf in postings: # iterate through posting list
			# accumulate cosine sim. using tf-idf wts, divide by doc vec length
			sims[docID] += tf*idf * qDct[tk]*idf / docLen[docID]
	return sims # return simularity scores of each document (most are 0)
	
def getTopNSimDocs(qID, simScore, N=100): # return top N document for a query
	topN = simScore.most_common(N) # use binary heap for extracting top N
	fmt = '%d Q0 %d %d %f Wu\n' # format for output file lines
	return [fmt % (qID,docID,n+1,score) for n,(docID,score) in enumerate(topN)]


with open(dictFileName, 'rb') as h:
	d = pickle.load(h) # load dictionary containing offset, DF, and idf
docLen = d['#docLen#'] # document vector length
print("Parsing query file: %s"%queryFileName)
print("\tQuerying inverted file %s and dict %s"%(invFileName,dictFileName))
with open(outFileName, 'w') as fh: # process each query while writing out
	for qInd,qTxt in enumerate(getQueries( queryFileName )): # loop over queries
		qDict = parseQuery(qTxt) # parse query into tokens and counts
		simScores = cosineSimScore(qDict) # calculate cosine sim for all docs
		out = getTopNSimDocs(qInd+1, simScores) # get top N docs based on sim
		fh.writelines(out) # write out lines for output
print("Wrote out query results to: %s" % outFileName)

