import sys, re, nltk, string

tg = re.compile('<p id=\d+>') # regex to match leading tag
pct = re.compile('[' + string.punctuation + ']+') # match 1+ consec. punctuation

def getParagraphs(fname):
	with open (fname, 'r') as f:
		rawStr = f.read().replace('\n','').lower() # to low case and remove LF
		rawStr = tg.sub('', rawStr) # remove lead tag
		rawStr = rawStr.split('</p>') # split by end tag
	return [x for x in rawStr if x] # return non-empty paragraphs

def processDoc(rawTxt):
	d = dict() # dictionary fodr this document
	tokens = nltk.word_tokenize(rawTxt) # split into tokens
	for tk in tokens:
		if not pct.match(tk): # if token is not (1+ consec.) punctuations
			d[tk] = d.get(tk,0) + 1 # +1 count for word in current document
	return d
	
def processFile(fname):
	print('Processing file: %s'%fname)
	paragraphs = getParagraphs(fname)
	
	corp = dict() # dictionary for entire collection
	for doc in paragraphs: # loop over all document
		d = processDoc(doc) # get dictionary of word and freq for current doc
	
		for wd in d: # loop over words in current doc
			tmp = corp.get(wd, (0,0)) # get collection freq and document freq
			corp[wd] = (tmp[0]+d[wd], tmp[1]+1) # sum coll. freq and +1 doc freq

	print('\tProcessed %d paragraphs'%len(paragraphs))
	print('\tVocabulary size: %d'%len(corp))
	print('\tCollection size: %d'%sum([corp[x][0] for x in corp]))
	return corp

def printCorpusInfo(corp):
	# reverse sort corpus by frequency and print 1-100, 500, 100, and 5000 rank
	revSrt = sorted(corp.items(), key=lambda kv: kv[1], reverse=True) 
	for n in list(range(100)) + [499, 999, 4999] :
		k = revSrt[n]
		print("\t#%d: '%s' - coll freq=%d, doc freq=%d"% ( (n+1,k[0])+(k[1])) )
	
	# print number of 
	z = len([x for x in revSrt if x[1][1]==1])
	print('\t%d words are in just 1 doc, %f%% of total'%(z, z/len(corp)*100))


######################################
# Parsing file and print corpus info
####################################### 
if len(sys.argv) == 2:
	corpus = processFile(sys.argv[1])
	printCorpusInfo(corpus)
else:
	print("Usage: %s [fileName]"%sys.argv[0])
