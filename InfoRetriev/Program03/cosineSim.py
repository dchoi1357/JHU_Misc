import sys, pickle, struct

def parseQueryFile(fname):
	with open(fname, 'r') as f:
		for line in f:
			tmp = line[:-1].lower().split(',')
			lkup.append( (int(tmp[0]), tmp[1:]) )
	return lkup

def readBinFileIntoInts(fName, offset, nInts):
	with open(fName, 'rb') as f:
		f.seek(offset*4)
		tmp = [x[0] for x in struct.iter_unpack('<I', f.read(4*2*nInts))]
	return list(zip(tmp[0::2], tmp[1::2]))

def lookupSingleTerm(term):
	df,offset = d[term]
	posting = readBinFileIntoInts(sys.argv[2], offset, df)
	return (df, posting)

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
	with open(sys.argv[3], 'rb') as h:
		d = pickle.load(h)
	lookups = readLookupFile(sys.argv[1])
	for x in lookups:
		lookupAndPrint(x[0], x[1])

else:
	print("Usage: %s [lookupInput] [invFileName] [dictName]"%sys.argv[0])
