import nltk, string, re
punct = re.compile('^['+string.punctuation+']+$') # match 1+ consec. punctuation
engStopWords =  set(nltk.word_tokenize( \
	' '.join(nltk.corpus.stopwords.words('english')) ) )

def tokenizeNoPunct(txt):
	return [tk for tk in nltk.word_tokenize(txt) if not punct.match(tk)]

def tokenizeNoPunctStopword(txt):
	return [tk for tk in nltk.word_tokenize(txt) if \
		(tk not in engStopWords and not punct.match(tk))]

def tokenizeNoPunctNStem(txt, n=5):
	return [tk[:5] for tk in nltk.word_tokenize(txt) if not punct.match(tk)]

def tokenizeNoPunctStopwordNStem(txt, n=5):
	return [tk[:5] for tk in nltk.word_tokenize(txt) if \
		(tk not in engStopWords and not punct.match(tk))]

def tokenize(txt, opt):
	if type(opt)==str:
		opt = int(opt)
	if opt == 0:
		return tokenizeNoPunct(txt)
	elif opt == 1:
		return tokenizeNoPunctNStem(txt)
	elif opt == 2:
		return tokenizeNoPunctStopword(txt)
	elif opt == 3:
		return tokenizeNoPunctStopwordNStem(txt)
	else:
		raise KeyError("No such options for %d."%opt)
