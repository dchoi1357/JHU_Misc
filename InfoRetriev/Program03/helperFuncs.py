import nltk

def getStopWordTokens():
	z = nltk.corpus.stopwords.words('english')
	return set(nltk.word_tokenize( ' '.join(z) ))

def tokenizeNoPunct(txt, punctRegex):
	return [tk for tk in nltk.word_tokenize(txt) if not punctRegex.match(tk)]

def tokenizeNoPunctStopword(txt, punctRegex, stopWordsSet):
	return [tk for tk in nltk.word_tokenize(txt) if \
		(tk not in stopWordsSet and not punctRegex.match(tk))]
