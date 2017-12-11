import nltk, textwrap
import numpy as np
#nltk.download('reuters')
from nltk.corpus import reuters


### Get all reuters training articles
fIDs = np.array(reuters.fileids())
trainIdx = np.where(np.array([ t[:4] for t in fIDs ]) == 'trai')[0]

nCatgs = np.empty(len(trainIdx), dtype=int)
catgs = np.empty(len(trainIdx), dtype=object) # pre-allocate
texts = np.empty(len(trainIdx), dtype=object) # pre-allocate
outNames = np.empty(len(trainIdx), dtype='<S16') # pre-allocate
for n,idx in enumerate(trainIdx):
	name = fIDs[idx]
	cat = reuters.categories(name)
	if cat: # if category is not empty
		tmp = name.split('/')
		outNames[n] = tmp[0] + '-' + tmp[1].zfill(5)
		nCatgs[n] = len( reuters.categories(name) )
		catgs[n] = ','.join( reuters.categories(name) )
		texts[n] = ' '.join( reuters.words(name) )

# trimming articles without categories
toTrim = np.invert(np.equal(catgs,None))
catgs = catgs[toTrim]
texts = texts[toTrim]
nCatgs = nCatgs[toTrim]
outNames = outNames[toTrim]

for n in range( len(outNames) ):
	with open('./reuters/' + outNames[n] + '.txt', 'w') as f:
		f.writelines('\n'.join( textwrap.wrap(texts[n], 80) ))

out = np.vstack((outNames, catgs)).T
out = out[ np.argsort(out[:,0]) ]
np.savetxt('reuters_catgs.csv', out, fmt='%s', delimiter=',')


## Save inaugural addresses
#nltk.download('inaugural')
from nltk.corpus import inaugural as inaug
adds = inaug.fileids()

texts = np.empty(len(adds), dtype=object) # pre-allocate
for n,name in enumerate(adds):
	texts[n] = ' '.join( inaug.words(name) )
	with open('./inaugural/' + name, 'w') as f:
		tmp = textwrap.wrap(texts[n], 80)
		f.writelines( '\n'.join(tmp).encode('ascii','ignore') )
	
	
	
	
	
