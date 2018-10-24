import sys, nltk
import tokenHelper as tkn # import custom function for tokenization of text
import numpy as np

d = 'E:\\VirtualMachines\\shared\\tmp\\'
trainFile = 'E:\\VirtualMachines\\shared\\tmp\\phase1.train.shuf.tsv'
test = 'E:\\VirtualMachines\\shared\\tmp\\test.tsv'

header = ['assessment','docID','title','authors','journal','ISSN','year', 
		'language', 'abstract', 'keywords']
convFun = {0: lambda s: int(s)>0, 1: lambda s: s[5:]}
tmp = np.genfromtxt(trainFile, delimiter='\t', dtype=None, names=header, 
					comments=None, encoding='utf-8', converters=convFun)