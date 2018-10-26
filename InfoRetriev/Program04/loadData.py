import sys, nltk, os
import tokenHelper as tkn # import custom function for tokenization of text
import numpy as np

#d = 'E:\\VirtualMachines\\shared\\tmp\\'
#trainFile = 'E:\\VirtualMachines\\shared\\tmp\\phase1.train.shuf.tsv'
#test = 'E:\\VirtualMachines\\shared\\tmp\\test.tsv'

d = os.path.join('c:', 'data', 'JHU', 'InfoRetriev', 'Program04')
trainFile = os.path.join(d, 'phase1.train.shuf.tsv')
testFile = os.path.join(d, 'phase1.test.shuf.tsv')

#trainFile = './data/phase1.train.shuf.tsv'

header = ['assessment','docID','title','authors','journal','ISSN','year',
		'language', 'abstract', 'keywords']

convFun = {0: lambda s: int(s)>0, 1: lambda b: b[5:]}
raw = np.genfromtxt(trainFile, delimiter='\t', dtype=None, names=header,
					comments=None, converters=convFun, encoding='utf-8')

