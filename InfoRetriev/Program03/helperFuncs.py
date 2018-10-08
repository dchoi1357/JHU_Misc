import nltk, string, re
#pct = re.compile('^['+string.punctuation+']+$') # match 1+ consec. punctuation

def getStopWordTokens():
	z = nltk.corpus.stopwords.words('english')
	return set(nltk.word_tokenize( ' '.join(z) ))

def tokenizeNoPunct(txt, punct):
	return [tk for tk in nltk.word_tokenize(txt) if not punct.match(tk)]

def tokenizeNoPunctStopword(txt, punctRegex, stopWordsSet):
	return [tk for tk in nltk.word_tokenize(txt) if \
		(tk not in stopWordsSet and not punctRegex.match(tk))]

