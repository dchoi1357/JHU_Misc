import numpy as np
from DecisionTree import DTnode

def Entropy(array):
    ''' Given an array, calculate the entropy of the elements.
    Using the counts of the unique elements, the function calculats entropy as
    sum( Pr(i) * log2(Pr(i))
    '''
    counts = np.unique(array, return_counts=True)[1]
    probs = counts / counts.sum()
    return -(probs*np.log2(probs)).sum()

def getSplitPoints(data,labels):
    ''' Get split points for continuous features.
    Given an array of feature samples and labels, the function returns all
    possible split points for candidate split points in building decision tree.

    Array and labels are sorted by ascending order of data. The split points are
    midpoints of all data where labels are different between the two points.
    '''
    srtIdx = np.argsort(data) # get sorted index for data vector
    srtdData = data[srtIdx] # data in sorted order
    midpoints = (srtdData[:-1] + srtdData[1:])/2 # midpoints between data pts
    srtdLabls = labels[srtIdx] # rearrange labels by sorted data order
    diffLabel = srtdLabls[:-1] != srtdLabls[1:] # find midpt where labels changed
    return midpoints[diffLabel] # return midpts where labels are different

def getBestSplitInfo(data, labels, splitPts):
    ''' Get the split point that would result in the lowest mean entropy.
    Given an input of data features, labels, and split points, the function 
    iterates through all split points, calculate the mean entropy of the 
    resulting halves, and pick out the best split point.
    For the best split point, calculate the intrinsic value of the split. 
    Returns the entropy, split point, intrinsic value of the best split point.
    '''
    
    bestEntropy = np.Inf # highest possible entropy value
    bestPoint = None
    bestPr = -1
    for n,pt in enumerate(splitPts): # loop over all split points
        LT = data < pt # idx for where data less than split point
        prLT = sum(LT) / data.size # probability of split point
        ent = prLT*Entropy(labels[LT]) + (1-prLT)*Entropy(labels[~LT]) # entropy
        if ent < bestEntropy: # record info if current entropy lower than best
            bestEntropy = ent
            bestPoint = pt
            bestPr = prLT
            
    if np.abs(bestPr-0) < np.finfo(bestPr.dtype).eps: # if homogenous data
        intrInfo = 0 # intrinsic info is 0
    else: # otherwise calculate instrinsic value, which is two-class entropy
        intrInfo = -bestPr*np.log2(bestPr) - (1-bestPr)*np.log2(1-bestPr)
    return bestEntropy,bestPoint,intrInfo

def SplitInfo(xs, ys):
    ''' Given samples of a feature and class labels, calculate various 
    statistics of splitting the sample.

    Given samples of a feature and class labels, the function will split the 
    sample and calculates and return mean entropy, intrinsic entropy value of
    the feature, and the split point (for numeric features only).
    '''
    if np.issubdtype(xs.dtype, np.number): # numeric features
        splitPts = getSplitPoints(xs, ys) # get all split points
        meanEnt,splitPt,intrinsVal = getBestSplitInfo(xs, ys, splitPts)
    else: # categorical features
        vals, Ns = np.unique(xs, return_counts=True) # counts of unique values
        # loop over all unique values, calculat entropy
        meanEnt = sum(Ns/len(ys) * [Entropy(ys[xs==v]) for v in vals])
        intrinsVal = Entropy(xs) # intrinsic value of the feature sample
        splitPt = None # no split point since not numeric
    return meanEnt,intrinsVal,splitPt

def selectBestFeature(data, labels, useRatio=True):
    ''' Select best feature to split sample on for best entropy gain ratio.

    Given input samples of multiple features and labels, the function will 
    iterate through all features, split the labels based on the feature (either
    numeric or categorical), and find the feature the produce the biggest
    entropy gain. The best feature, gain ratio, split point, and the info gain
    associated with this split is returned.
    '''
    features = data.dtype.names # all feature names
    info = Entropy(labels) # base entropy of labels prior to split
    gains = np.empty(len(features)) # pre-allocate
    gainRatios = np.empty(len(features))
    splitPts = [None] * len(features)
    for n,feat in enumerate(features): # for all features, test the split
        expEntropy,intrnVal,splitPts[n] = SplitInfo(data[feat], labels)
        gains[n] = info-expEntropy # info gain of the split
        gainRatios[n] = gains[n] / (0.01+intrnVal if useRatio else 1)
        #print("%s exp ent: %f"%(feat,gainRatios[n]))
    maxN = np.argmax(gainRatios) # find feature with largest gain ratio
    return features[maxN],gainRatios[maxN],splitPts[maxN],gains[maxN]

def TrainDTree(allData, allLabels, minGain=0):
    ''' Train decision tree

    Given a set of features and associated samples and labels, the fuction will
    train a decision tree with C4.5 algorithm. The algorithm is applied 
    recursively in a depth-first manner. The start of the algorithm is the 
    entire data set. At every split, a subset of the data set and labels are
    carried on to the child nodes. In order to avoid stack overflow, only the
    index of the data is passed onto the recursion, while only one copy of 
    the data set is kept in memory.
    '''
    def c4_5(idx, featureSet, defLabel):
        if sum(idx) == 0:# empty data, class = default label
            return DTnode(defLabel) # node with default labels
        data,labels = allData[idx][list(featureSet)],allLabels[idx] # relv. data
        
        (values,counts) = np.unique(labels, return_counts=True) # counts of uniq
        majority = str(values[np.argmax(counts)]) # get majority class as default
        if len(counts)==1 or not featureSet: # homogenous or no attribs
            return DTnode(majority) # node with majority labels
        
        # select best feature
        bestFeat,gainRatio,splitPt,gain = selectBestFeature(data,labels)
        #print('GainR=%f, Gain=%f'%(gainRatio,gain))
        if gain < minGain: # early stopping if gain < defined thresh
            return DTnode(majority)
        
        featSubset = featureSet - set([bestFeat]) # remove used feat from algo
        # construct tree node with feature and split point, also majority label
        node = DTnode(bestFeat, splitPt, majority) 
        if splitPt is None: # no split point, categorical feature
            for val in set(data[bestFeat]): # loop over features
                subIdx = idx[data[bestFeat] == val] # index of sub-nodes data
                child = c4_5(subIdx, featSubset, majority) # recur into child
                node.addChild(child, val) # add child node to current node
        else: # numerical feature, 2 child nodes
            lessThan = data[bestFeat] < splitPt # idx of data less than
            child = c4_5(idx[lessThan], featSubset, majority) # recur data <
            node.addChild(child, '<') # add child node to current node
            child = c4_5(idx[~lessThan], featSubset, majority) # recur data >
            node.addChild(child, '>=') # add child node to current node
        return node

    uniqLabels,uniqCounts = np.unique(allLabels, return_counts=True)
    labelMajority = uniqLabels[np.argmax(uniqCounts)] # most common label
    allFeatures = set(allData.dtype.names) # set of all features
    allIdx = np.arange(allData.size) # numeric idx of all rows
    return c4_5(allIdx, allFeatures, labelMajority) # root of DTree