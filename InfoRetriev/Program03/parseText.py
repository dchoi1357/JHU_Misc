import sys
if len(sys.argv) != 4: # check number of commandline inputs
	print("Usage: %s [inputFile] [outInvFileName] [outDictName]"%sys.argv[0])
	exit()

import nltk, re, string, pickle
from operator import itemgetter
import helperFuncs as funcs

pct = re.compile('^['+string.punctuation+']+$') # match 1+ consec. punctuation
tg = re.compile(r'<p id=(\d+)>', re.IGNORECASE) # regex to capture docID
vcb = dict() # dictionary for entire collection
nDocs = 0 # number of documents processed

def processDocsFile(fname):
	global nDocs, vcb
	inDoc = False # whether currently in the middle of parsing a Doc
	docID = 0 # the current document ID the program is in
	
	with open (fname, 'r') as f: # parse through file
		for line in f: # NOTE: must read file line by line due to large size
			if not inDoc: # before start of a document
				tagMatches = tg.findall(line) # regex matching lead tag
				if tagMatches: # regex match lead tag
					docID = int( tagMatches[0] ) # first match as docID
					inDoc = True
					tmpTxts = list() # accumulate for processing doc by doc
			else: # in the middle of parsing a document
				tmp = line.strip().casefold()
				if tmp == '</p>': # end of document
					inDoc = False # set to indicate not in middle of document
					nDocs += 1 # increment document count
					tmpDict = processDoc(' '.join(tmpTxts), docID) # process doc
				else:
					tmpTxts.append( tmp ) # add to list of lines for doc
					
	for term in vcb: # go through dict and sort the posting lists
		vcb[term].sort(key=itemgetter(0)) # sort by first elem, or docID

def processDoc(txt, docid):
	global vcb
	d = dict() # temp dict to store info on this document
	for tk in funcs.tokenizeNoPunct(txt, pct):
		d[tk] = d.get(tk, 0) + 1
	for tk in d: # merge dict of this doc with the bigger vocab dict
		if tk not in vcb: # if not in vocab
			vcb[tk] = [(docid, d[tk])] # add first posting 
		else: # if already in vocab
			vcb[tk].append( (docid, d[tk]) ) # append to posting list
	return d


def writeInvertedFile(invFileName, dictFileName):
	global vcb, nDocs
	outDict = dict() # dictionary to be written out
	count, coll = 0,0 # running offset count and collection size
	with open(invFileName, 'wb') as f:
		for term in vcb:
			outDict[term] = (len(vcb[term]), count)
			count += 2*len(vcb[term]) # increment current count
			for docid,n in vcb[term]: # write tuple of docid and tf(term,docid)
				coll += n
				f.write( docid.to_bytes(4, byteorder='little') )
				f.write( n.to_bytes(4, byteorder='little') )

	outDict['#nDocs#'] = nDocs # add entry for number of documents
	with open(dictFileName, 'wb') as h: # write dic out as pickle file
		pickle.dump(outDict, h, protocol=pickle.HIGHEST_PROTOCOL)
	return outDict, coll


def processFile(fname):
	print('Processing file: %s'%fname)
	processDocsFile(fname)
	# write inverted file and dictionary
	_,coll = writeInvertedFile(sys.argv[2], sys.argv[3]) 
	print('\tProcessed %d documents'%nDocs)
	print('\tVocabulary size: %d'%len(vcb))
	print('\tCollection size: %d'%coll)

	print('Wrote inverted file %s, dict file %s.' % (sys.argv[2],sys.argv[3]) )


processFile(sys.argv[1])

