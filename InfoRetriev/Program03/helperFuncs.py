import nltk, string, re
punct = re.compile('^['+string.punctuation+']+$') # match 1+ consec. punctuation
engStopWords =  set(nltk.word_tokenize( \
	' '.join(nltk.corpus.stopwords.words('english')) ) )

def tokenizeNoPunct(txt):
	return [tk for tk in nltk.word_tokenize(txt) if not punct.match(tk)]

def tokenizeNoPunctStopword(txt):
	return [tk for tk in nltk.word_tokenize(txt) if \
		(tk not in engStopWords and not punct.match(tk))]

