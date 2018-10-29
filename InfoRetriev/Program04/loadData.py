import sys, nltk, os, sklearn
import tokenHelper as tkn # import custom function for tokenization of text
import numpy as np

d = os.path.join('e:', 'VirtualMachines', 'shared', 'tmp')
# d = os.path.join('c:', 'data', 'JHU', 'InfoRetriev', 'Program04')
trainFile = os.path.join(d, 'phase1.train.shuf.tsv')
devFile = os.path.join(d, 'phase1.dev.shuf.tsv')
testFile = os.path.join(d, 'phase1.test.shuf.tsv')

header = ['assessment','docID','title','authors','journal','ISSN','year',
		'language', 'abstract', 'keywords'] # the variable names for file

def readFile(fName):
	convFun = {1: lambda b: b[5:]} # to capture hash code without prefix
	return np.genfromtxt(fName, delimiter='\t', dtype=None, names=header,
		comments=None, converters=convFun, encoding='utf-8') # load file

raw = readFile(trainFile)
binary = sklearn.preprocessing.LabelBinarizer(pos_label=1) # 1 = positive class
labels = binary.fit_transform(raw['assessment']).ravel()  # binarize assessments

# The following code tokenizes input text using TF-IDF and get features
from sklearn.feature_extraction.text import TfidfVectorizer
vectorize = TfidfVectorizer(tokenizer=tkn.tokenizeNoPunctStopword)
x = vectorize.fit_transform( raw['title'] ) # extract features

from sklearn import naive_bayes as NB 
mn_nb = NB.MultinomialNB().fit(x, labels)
bern_nb = NB.BernoulliNB().fit(x, labels)
#comp_nb = NB.Compl