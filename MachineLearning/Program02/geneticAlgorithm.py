import numpy as np, pandas as pd
from functions import Silhouette, pairwiseDist
from kmeans import kMeans

def evalFitness(dataMat, k, pop, preEval, dist):
	''' Evaluate fitness of the entire population of feature sets.
	Loop over every feature set in the population, and for every features set
	(individual), check a dictionary of evaluated fitness scores to see if the
	feature set has been evaluated already. If so, use memoized results. If not,
	calculate Silouette coefficient and add 1 to the coefficient as the fitness
	score, and then save this result to the memo dictionary.
	'''
	fitness = np.empty(pop.shape[0]) # store fitness of individuals
	for n,indv in enumerate(pop): # loop over populations one by one
		gene = ''.join(['1' if x else '0' for x in indv]) # string repr of DNA
		if gene in preEval: # combo of features previously evaluated
			fitness[n] = preEval[gene] # recall from dict
		else: # never evaluated before
			means,labels = kMeans(dataMat[:,indv], k) # cluster w/ features
			fitness[n] = Silhouette(dataMat,labels,dist).mean()+1 # fit > 0
			preEval[gene] = fitness[n] # store into dict for memoization
	return fitness,preEval

def selectParents(fitness, popSize):
	'''Use fitness proportional selection to select parents for next generation.
	Fitness is 1 + Silhouette coefficient, which are always positive. The 
	probability of being chosen as a parents is proportional to the individual's
	fitness score. Possible to be chosen twice.
	'''
	cumFit = np.cumsum(fitness)/np.sum(fitness) # cum array of normalized fitness
	rands = np.random.rand(popSize) # uniform random between 0,1
	outInd = np.searchsorted(cumFit, rands) # higher prob of select high fitness
	return outInd

def crossOver(pop, parentIdx):
	'''Perform cross-over to mix the DNA of parents for next generation.
	Receives input of the population of feature sets and the indices of 
	individuals chosen as parents. For every pair of parents, choose a random
	cross-over point where the first child contains the first half of the 
	father, the second half of the mother, and the second child contains
	the first half of the DNA of teh mother, and the second half of the father.

	With the next generation, ensures that no feature sets has no features 
	selected. Returns the new population after cross-over. 
	'''
	popN = pop.shape[0]
	idxDad = parentIdx[:len(parentIdx)//2] # first half of selected
	idxMom = parentIdx[len(parentIdx)//2:] # second half of selected
	breakPts = np.random.randint(1,pop.shape[1],popN//2) # x-over points

	out = np.empty(pop.shape, bool) # pre-allocate array for next gen
	for n,(d,m) in enumerate(zip(idxDad,idxMom)): # loop over parents and cross
		out[n] = np.hstack([ pop[d,:breakPts[n]],pop[m,breakPts[n]:] ])
		out[popN-n-1] = np.hstack([ pop[m,:breakPts[n]],pop[d,breakPts[n]:] ])
	out = minOneFeature(out) # all individuals must have 1 feature chosen
	return out

def mutate(pop, prob):
	''' Performs mutation operator for population with a probability.
	For every individual, probabilistically choose whether to apply mutation. 
	If an individual is chosen, randomly choose one bit of the DNA to flip by
	flipping the bit to exclude if originally include and vice versa.

	Ensures that all resulting individual sets has one feature chosen.
	'''
	toMutate = np.where(np.random.rand(pop.shape[0])<prob)[0] # indv to mutate
	mutatePts = np.random.randint(0,pop.shape[1],len(toMutate)) # locations
	for idx,n in zip(toMutate,mutatePts): # mutate selected individuals
		pop[idx,n] = ~pop[idx,n] # flip the selection bit
	pop = minOneFeature(pop) # all individuals must have 1 feature chosen
	return pop

def minOneFeature(pop): 
	'''Ensures at least 1 feature is selected in every individual.
	Since each individual corresponds to a set of features to use for 
	clustering, every individual must have one feature chosen. The function
	loops over all individuals in population, and if an individual has no 
	feature chosen, randomly flips one to on.
	'''
	noFeatIdx = pop.sum(axis=1)==0 # data pts with no features selected
	for n in np.where(noFeatIdx)[0]: # loop over all data with no features
		pop[n, np.random.randint(pop.shape[1])] = True # randomly select 1
	return pop
    
def geneticAlgoSelect(data, k, prm, trace=False):
	'''Main function of genetic algorithm selection.
	Generate a population of feature sets by randomly generating 0 and 1s 
	given a probability as specified in the input parameter.

	For this population, evaluate the fitness with the help of a memo. By 
	storing computation results into a dictionary, subsequent individuals with
	the same set of features can be skipped and results retrieved from the dict.

	The algorithm is to considered have converged if improvement to Silhouette
	coefficient has not been made for a specific number of generations. This
	minimum required improvement is specified in the input parameter. For 
	every generation, performs the regular selection, crossover, and mutation
	operator on the population.
	'''
	pop = np.random.rand(prm['popSize'],data.shape[1]) < prm['onProb']
	pop = minOneFeature(pop) # at least 1 feature must be selected
	memo = dict() # dict of result for memoization
	dMat = pairwiseDist(data) # pre-calc distance matrix for memoization

	baseFit = 0 # worst possible fitenss score
	converged,gen,stagnGens = False,1,0 # initialize loop vars
	while not converged: # loop until GA has converged
		#print(np.asanyarray(pop,int))
		fit,memo = evalFitness(data, k, pop, memo, dMat) # evaluate fitness
		bestIdx = np.argmax(fit) # keep track of best indiviaul
		bestFit,bestIndv = fit[bestIdx],pop[bestIdx] # best fit and features
		#print((bestFit,np.where(bestIndv)[0]))

		if (bestFit-baseFit < prm['minImprove']) and stagnGens>prm['stagnLim']:
			converged = True
			out = baseFit-1,np.where(baseIndv)[0] # silhouette coeff and list
		else: # not converged, selection + crossover + mutation
			if (bestFit-baseFit < prm['minImprove']):
				stagnGens += 1
			else:
				baseFit,baseIndv = bestFit,bestIndv # record long-run best
			parentInd = selectParents(fit, pop.shape[0]) # select parents
			pop = crossOver(pop, parentInd) # cross-over to get next gen
			pop = mutate(pop,prm['mutateProb']) # mutate

		if trace:
			print('Generation %d: best fitness = %.10f'%(gen,baseFit))
			print('\tBest set: %s'%str(np.where(baseIndv)[0]))
		gen += 1
	return out