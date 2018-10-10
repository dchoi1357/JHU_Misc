# This module contains different methods for tokenizing a document of text
# these are used in both parts of the program

import nltk, string, re
punct = re.compile('^['+string.punctuation+']+$') # match 1+ consec. punctuation
# list of stop words in English, tokenized by word_tokenize()
engStopWords =  set(nltk.word_tokenize( \
	' '.join(nltk.corpus.stopwords.words('english')) ) ) 

def tokenizeNoPunct(txt): # tokenize and take away all 1+ consec. punctuations
	return [tk for tk in nltk.word_tokenize(txt) if not punct.match(tk)]

def tokenizeNoPunctStopword(txt): # tokenize, no 1+ consec punct and stop words
	return [tk for tk in nltk.word_tokenize(txt) if \
		(tk not in engStopWords and not punct.match(tk))]

def tokenizeNoPunctNStem(txt, n=5): # apply 5-stem and no 1+ consec. punctuation
	return [tk[:5] for tk in nltk.word_tokenize(txt) if not punct.match(tk)]

def tokenizeNoPunctStopwordNStem(txt, n=5): # 5-stem, no 1+ punct, no stop words
	return [tk[:5] for tk in nltk.word_tokenize(txt) if \
		(tk not in engStopWords and not punct.match(tk))]

def tokenize(txt, opt): # Switch for deciding which method to use based on
	if type(opt)==str:  # another parameter. Can be set via cmd line argument
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
