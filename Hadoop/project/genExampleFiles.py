import random, textwrap, bisect, itertools

fruits = ['apple', 'banana', 'orange', 'grape', 'strawberry']
countries = ['England', 'China', 'Japan', 'France', 'Russia']
furnitures = ['bed', 'chair', 'lamp', 'table', 'sofa']
animals = ['dog', 'cat', 'fish', 'bird', 'lion', 'snake']
clothes = ['shirt', 'pants', 'jacket', 'hat', 'sweater']
terms = fruits + countries + furnitures + animals + clothes
with open('SimpleWords400.dat') as f:
	texts = f.read().splitlines()

punctuations = [' ', ', ', '. ']

def cumsum(l):
	total = 0
	cumProbs = list()
	for p in l:
		total += p
		cumProbs.append(total)
	return cumProbs

probs = cumsum([0.8, 0.1, 0.1]) # texts, terms, specific terms
puncts = cumsum([0.8, 0.1, 0.1]) # prob of space vs comma vs period

wordCt = [300, 500]

def genDoc(spec):
	corpus = [texts, terms, spec]
	idx = [bisect.bisect(probs, random.random()) 
		for x in range(random.randrange(wordCt[0], wordCt[1]))]
	words = [random.choice(corpus[i]) for i in idx]
	pt = [punctuations[bisect.bisect(puncts, random.random())] 
		for x in range(len(idx)-1)]
	
	l = list(words[0]) + \
		list(it.next() for it in itertools.cycle([iter(pt),iter(words)]))

	t = ''.join(l)
	return textwrap.wrap(t, 80)


def printListOfTxt(l,name):
	with open(name, 'w') as f:
		f.writelines('\n'.join(l))

printListOfTxt(genDoc(fruits), 'sample_fruits.txt')
printListOfTxt(genDoc(countries), 'sample_countries.txt')
printListOfTxt(genDoc(furnitures), 'sample_furnitures.txt')
printListOfTxt(genDoc(animals), 'sample_animals.txt')
printListOfTxt(genDoc(clothes), 'sample_clothes.txt')
