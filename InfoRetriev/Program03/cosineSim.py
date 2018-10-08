import sys, pickle, struct, math
import helperFuncs as funcs
from collections import Counter

tg = re.compile('<q id=\\d+>') # regex to match leading tag
nDocs = 0
vcb = dict() # corpus to be added as queries are looked up

def getQueries(fname):
	with open (fname, 'r') as f:
		rawStr = f.read().casefold() # to low case and remove LF
		rawStr = tg.sub('', rawStr) # remove lead tag
		rawStr = rawStr.split('</q>') # split by end tag
	return [x for x in rawStr if x.strip()] # return non-empty paragraphs

def parseQuery(qtxt):
	return Counter(funcs.tokenizeNoPunct(qtxt))

def readBinFileIntoInts(fName, offset, nInts):
	with open(fName, 'rb') as f:
		f.seek(offset*4)
		tmp = [x[0] for x in struct.iter_unpack('<I', f.read(4*2*nInts))]
	return list(zip(tmp[0::2], tmp[1::2]))

def lookupPostingIDF(term):
	df,offset,idf = d[term]
	posting = readBinFileIntoInts(invFilePath, offset, df)
	return ( posting, idf )

def cosineSim(qDct):
	qVecLen = 0
	sims = Counter() # counter for storing simularity scores
	for tk in qDct: 
		if tk not in vcb: # if have not been looked up before
			vcb[tk] = lookupPostingIDF(tk) # lookup posting from inv file
		postings,idf = vcb[tk] # get posting list and IDF for term
		for docID,tf in postings: # iterate through posting list
			sims[docID] += tf*idf * qDct[tk]*idf / docLen[docID-1]
	return sims
	
def getTopNSimDocs(qID, simScore, N=100):
	topN = simScore.most_common(N)
	fmt = '%d Q0 %d %d %f Wu'
	return [fmt % (qID,docID,n,score) for n,(docID,score) in enumerate(topN)]

def lookupAndPrint(srchTyp, terms):
	terms = terms[0] if len(terms)==1 else terms # if only 1 term, then unnest
	try:
		if srchTyp == 0:
			df, pst = lookupSingleTerm(terms)
			print("%s: df=%d"%(terms,df))
		elif srchTyp == 2:
			df, pst = lookupSingleTerm(terms)
			print("%s: df=%d, postings=%s"%(terms,df,str(pst)))
		elif srchTyp == 1:
			_, pst1 = lookupSingleTerm(terms[0]) # posting list for term 1
			_, pst2 = lookupSingleTerm(terms[1]) # posting list for term 2
			# give sorted list of intersection of doc ids
			ids = sorted( set([x[0] for x in pst1]) & set([x[0] for x in pst2]) )
			print("docs with %s & %s: %s"%(terms[0],terms[1],str(ids)))
	except KeyError:
		print('%s is not within corpus.'%str(terms))
		

if len(sys.argv) == 4:
	invFilePath = sys.argv[2]
	with open(sys.argv[3], 'rb') as h:
		d = pickle.load(h)
	nDocs = d['#nDocs#']
	lookups = readLookupFile(sys.argv[1])
	for x in lookups:
		lookupAndPrint(x[0], x[1])

else:
	print("Usage: %s [lookupInput] [invFileName] [dictName]"%sys.argv[0])
