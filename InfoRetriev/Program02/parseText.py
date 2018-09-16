import sys, nltk, re, string, struct, pickle

tg = re.compile('<p id=\\d+>') # regex to match leading tag
pct = re.compile('[' + string.punctuation + ']+') # match 1+ consec. punctuation
vcb = dict() # dictionary for entire collection

def getParagraphs(fname):
	with open (fname, 'r') as f:
		rawStr = f.read().replace('\n','').lower() # to low case and remove LF
		rawStr = tg.sub('', rawStr) # remove lead tag
		rawStr = rawStr.split('</p>') # split by end tag
	return [x for x in rawStr if x] # return non-empty paragraphs

def processDoc(rawTxt, docid):
	d = dict() # temp dict to store info on this document
	for tk in nltk.word_tokenize(rawTxt): # split into tokens
		if not pct.match(tk): # if token is not (1+ consec.) punctuations
			d[tk] = d.get(tk, 0) + 1
	for tk in d: # merge dict of this doc with the bigger vocab dict
		if tk not in vcb: # if not in vocab
			vcb[tk] = [(docid, d[tk])] # add first posting 
		else: # if already in vocab
			vcb[tk].append( (docid, d[tk]) ) # append to posting list
	return d

def writeInvertedFile(invFileName, dictFileName):
	outDict = dict() # dictionary to be written out
	count = 0 # running offset count
	with open(invFileName, 'wb') as f:
		for term in vcb:
			outDict[term] = (len(vcb[term]), count)
			count += 2*len(vcb[term]) # increment current count
			for docid,n in vcb[term]: # write tuple of docid and tf(term,docid)
				f.write( docid.to_bytes(4, byteorder='little') )
				f.write( n.to_bytes(4, byteorder='little') )

	with open(dictFileName, 'wb') as h: # write dic out as pickle file
		pickle.dump(outDict, h, protocol=pickle.HIGHEST_PROTOCOL)
	return outDict

def processFile(fname):
	print('Processing file: %s'%fname)
	paragraphs = getParagraphs(fname)
	for n,doc in enumerate(paragraphs): # loop over all document
		processDoc(doc, n+1) # parse each doc and update full dictionary
	writeInvertedFile(sys.argv[2], sys.argv[3]) # write inv file and dic

	print('Wrote inverted file %s, dict file %s.' % (sys.argv[2],sys.argv[3]) )


if len(sys.argv) == 4:
	processFile(sys.argv[1])
else:
	print("Usage: %s [fileName]"%sys.argv[0])
