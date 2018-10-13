import sys
if len(sys.argv) != 5: # check number of commandline inputs
	print(("Usage: %s [inputFile] [outInvFileName] [outDictName] " \
		"[tokenizeAlgoNumber]") %sys.argv[0])
	sys.exit()

import nltk, re, string, pickle, math
from operator import itemgetter
from collections import Counter
import tokenHelper as tkn # import custom function for tokenization of text

# command line arguments
inputFile,outInvFile,outDictFile,tokenizeAlgoNum = \
	[ sys.argv[n] for n in range(1,len(sys.argv)) ]
tg = re.compile(r'<p id=(\d+)>', re.IGNORECASE) # regex to capture docID
vcb = dict() # dict for entire collection, contains term and posting list
nDocs = 0 # number of documents processed

def processDocsFile(fname): # process the entire corpus file
	global nDocs, vcb
	inDoc = False # whether currently in the middle of parsing a Doc
	docID = 0 # the current document ID the program is in
	
	with open (fname, 'r') as f: # parse through file
		for line in f: # NOTE: must read file line by line due to large size
			if not inDoc: # before start of a document
				tagMatches = tg.findall(line) # regex matching lead tag
				if tagMatches: # regex match lead tag
					docID = int( tagMatches[0] ) # first match as docID
					inDoc = True # set to middle of parsing a doc
					tmpTxts = list() # accumulate to process entire doc at once
			else: # in the middle of parsing a document
				tmp = line.strip().casefold() # strip whitespace and lower case
				if tmp == '</p>': # end of document
					inDoc = False # set to indicate not in middle of document
					nDocs += 1 # increment document count
					tmpDict = processDoc(' '.join(tmpTxts), docID) # process doc
				else:
					tmpTxts.append( tmp ) # add to list of lines for doc
			
			if nDocs%100 == 0: # update progress every 100 doc processed
				print('\tDocument processed: %10d'%nDocs, end='\r')
				print('\b' * (10+len('Document processed: ')), end='\r')
		print('\tDocument processed: %10d'%nDocs) # print final nDocs processed

	for term in vcb: # go through dict and sort the posting lists
		vcb[term].sort(key=itemgetter(0)) # sort by first elem, or docID


def processDoc(txt, docid): # process single doc as long string of texts
	global vcb
	d = Counter( tkn.tokenize(txt,tokenizeAlgoNum) ) # count of each token
	for tk in d: # merge dict of this doc with the bigger vocab dict
		if tk not in vcb: # if not in vocab
			vcb[tk] = [(docid, d[tk])] # add first posting 
		else: # if already in vocab
			vcb[tk].append( (docid, d[tk]) ) # append to posting list
	return d


def writeInvertedFile(invFileName, dictFileName): # write inverted file as bin
	global vcb, nDocs
	outDict = dict() # dictionary to be written out
	count, coll = 0,0 # running offset count and collection size
	docLen = Counter() # must use dict as docID may not be contiguous
	with open(invFileName, 'wb') as f: # open binary file for writing
		for term in vcb: # loop over all terms in collection
			posts = vcb[term]
			idf = math.log(1.0 + nDocs/len(posts), 2) # +1.0 for term in all doc
			outDict[term] = (len(posts), count, idf) # save DF, offset, and idf
			count += 2*len(posts) # increment current count
			for docid,tf in posts: # write pair of docid and tf(term,docid)
				coll += tf # add tf of term to total collection size
				f.write( docid.to_bytes(4, byteorder='little') ) # 4-byte bin
				f.write( tf.to_bytes(4, byteorder='little') ) # 4-byte bin
				docLen[docid] += (tf*idf)**2 # accumulate doc vector length

	for docID in docLen: # loop over all docs to calculate proper doc vec length
		docLen[docID] = math.sqrt(docLen[docID]) # sqrt of sum of squared terms
	outDict['#nDocs#'] = nDocs # add entry for number of documents
	outDict['#docLen#'] = docLen # add entry for document vector lengths
	with open(dictFileName, 'wb') as h: # write dic out as pickle file
		pickle.dump(outDict, h, protocol=pickle.HIGHEST_PROTOCOL)
	return outDict, coll


def processFile(fname):
	print('Processing file: %s'%fname)
	processDocsFile(fname) # process entire text corpus
	# write inverted file and dictionary
	_,coll = writeInvertedFile(outInvFile, outDictFile) 
	print('\tVocabulary size: %d'%len(vcb))
	print('\tCollection size: %d'%coll)
	
	print('Wrote inverted file %s, dict file %s.' % (outInvFile,outDictFile) )

processFile(inputFile)

