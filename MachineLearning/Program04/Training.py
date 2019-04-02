import numpy as np
from DecisionTree import DTnode

def Entropy(array):
    counts = np.unique(array, return_counts=True)[1]
    probs = counts / counts.sum()
    return -(probs*np.log2(probs)).sum()

def getSplitPoints(data,labels):
    srtIdx = np.argsort(data) # get sorted index for data vector
    srtdData = data[srtIdx] # data in sorted order
    midpoints = (srtdData[:-1] + srtdData[1:])/2 # midpoints between data pts
    srtdLabls = labels[srtIdx] # rearrange labels by sorted data order
    diffLabel = srtdLabls[:-1] != srtdLabls[1:] # find midpt where labels changed
    return midpoints[diffLabel] # return midpts where labels are different

def getBestSplitInfo(data, labels, splitPts):
    bestEntropy = np.Inf
    bestPoint = None
    bestPr = -1
    for n,pt in enumerate(splitPts):
        LT = data < pt
        prLT = sum(LT) / data.size
        ent = prLT*Entropy(labels[LT]) + (1-prLT)*Entropy(labels[~LT])
        if ent < bestEntropy:
            bestEntropy = ent
            bestPoint = pt
            bestPr = prLT
            
    if (bestPr-0) < np.finfo(bestPr.dtype).eps: # if homogenous data, prob=0
        intrInfo = 0
    else:
        intrInfo = -bestPr*np.log2(bestPr) - (1-bestPr)*np.log2(1-bestPr)
    return bestEntropy,bestPoint,intrInfo

def SplitInfo(xs, ys):
    if np.issubdtype(xs.dtype, np.number): # numeric features
        splitPts = getSplitPoints(xs, ys)
        meanEnt,splitPt,intrinsVal = getBestSplitInfo(xs, ys, splitPts)
    else: # categorical features
        vals, Ns = np.unique(xs, return_counts=True)
        meanEnt = sum(Ns/len(ys) * [Entropy(ys[xs==v]) for v in vals])
        intrinsVal = Entropy(xs)
        splitPt = None
    return meanEnt,intrinsVal,splitPt

def selectBestFeature(data, labels, useRatio=True):
    features = data.dtype.names
    info = Entropy(labels)
    gains = np.empty(len(features))
    gainRatios = np.empty(len(features))
    splitPts = [None] * len(features)
    for n,feat in enumerate(features):
        expEntropy,intrnVal,splitPts[n] = SplitInfo(data[feat], labels)
        gains[n] = info-expEntropy
        gainRatios[n] = gains[n] / (0.01+intrnVal if useRatio else 1)
        #print("%s exp ent: %f"%(feat,gainRatios[n]))
    maxN = np.argmax(gainRatios)
    return features[maxN],gainRatios[maxN],splitPts[maxN],gains[maxN]

def TrainDTree(allData, allLabels, minGain=0):
    def c4_5(idx, featureSet, defLabel):
        if sum(idx) == 0:# empty data, class = default label
            return DTnode(defLabel)
        data,labels = allData[idx][list(featureSet)],allLabels[idx]
        
        (values,counts) = np.unique(labels, return_counts=True)
        majority = str(values[np.argmax(counts)]) # get majority class as default
        if len(counts)==1 or not featureSet: # homogenous or no attribs
            return DTnode(majority)
        
        bestFeat,gainRatio,splitPt,gain = selectBestFeature(data,labels)
        #print('GainR=%f, Gain=%f'%(gainRatio,gain))
        if gain < minGain: # early stopping if gain < defined thresh
            return DTnode(majority)
        
        featSubset = featureSet - set([bestFeat])
        node = DTnode(bestFeat, splitPt, majority)
        if splitPt is None: # no split point, categorical feature
            for val in set(data[bestFeat]):
                subIdx = idx[data[bestFeat] == val]
                child = c4_5(subIdx, featSubset, majority)
                node.addChild(child, val)
        else: # numerical feature, 2 child nodes
            lessThan = data[bestFeat] < splitPt
            child = c4_5(idx[lessThan], featSubset, majority)
            node.addChild(child, '<')
            child = c4_5(idx[~lessThan], featSubset, majority)
            node.addChild(child, '>=')
        return node

    uniqLabels,uniqCounts = np.unique(allLabels, return_counts=True)
    labelMajority = uniqLabels[np.argmax(uniqCounts)]
    allFeatures = set(allData.dtype.names) # set of all features
    allIdx = np.arange(allData.size) # numeric idx of all rows
    return c4_5(allIdx, allFeatures, labelMajority) # root of DTree