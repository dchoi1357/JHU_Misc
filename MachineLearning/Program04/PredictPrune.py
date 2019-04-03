import numpy as np

def PredictDTree(tree, data):
    ''' Predict the labels of the input data using a decision tree.

    Algorithm uses recursion in depth-first manner into the leaf node. At each
    node, data are split as according to the decision tree node. To prevent
    stack overflow, only one copy of output and data are kept. At each node
    the indices of the data subset are preserved.
    '''
    def classify(node, idx):
        ndData = data[idx] # portion of data for this node
        if node.isLeaf(): # if leaf node, then make prediction
            result[idx] = node.attrib
            return
        else:
            feat = node.attrib # feature relevant to current tree node
            if node.splitPoint is None: # categorical
                for k in node.getValues(): # loop over all value of variable
                    subIdx = idx[ ndData[feat]==k ] # data subset of child node
                    classify(node.getChild(k), subIdx) # recur classify
            else: # numeric, two classes
                # data subset of child node < and >= of the split point
                lessThan = ndData[feat]<node.splitPoint 
                classify(node.getChild('<'), idx[lessThan]) 
                classify(node.getChild('>='), idx[~lessThan])
            return

    allIdx = np.arange(data.size) # indices of all data points
    result = np.empty(data.size, object) # prediction result
    classify(tree, allIdx)
    return result
  
def PruneDTree(tree, data, actuals):
    ''' Prune decision tree using a pruning set.

    Algorithm uses recursion in depth-first manner into the leaf node. At each
    node, data are split as according to the decision tree node. To prevent
    stack overflow, only one copy of output and data are kept. At each node
    the indices of the data subset are preserved.

    At the leaf node, the correct number of predictions are recorded at the
    nCorr attribute of the DTnode class. At an inner node, the sum of all 
    correct predictions of the entire sub-tree is compared to classification
    result based on majority label seen in training set. If prediction is not
    better, than remove the sub-tree and make current node a child node with
    majority label as the prediction.

    The algorithm prunes the entire tree at once in a depth-first manner, 
    using recursion.
    '''
    def prune(node, idx):
        ndData,ndActs = data[idx],actuals[idx] # data and labels for the node
        if node.isLeaf():
            node.nCorr = sum(node.attrib==ndActs) # save nCorrect preds
            return
        else:
            feat = node.attrib # feature relevant to current tree node
            if node.splitPoint is None: # categorical
                for k in node.getValues(): # loop over all value of variable
                    subIdx = idx[ ndData[feat]==k ] # data subset of child node
                    prune(node.getChild(k), subIdx) # recursively prune
            else: # numeric, two classes
                # data subset of child node < and >= of the split point
                lessThan = ndData[feat]<node.splitPoint
                prune(node.getChild('<'), idx[lessThan])
                prune(node.getChild('>='), idx[~lessThan])
            
            node.nCorr = sum(node.getChildCorrNum()) # sum childNodes error nums
            nCorrNaive = sum(ndActs==node.preEval) # nCorr using label majority
            if nCorrNaive > node.nCorr: # if majority class is better than C4.5
                node.makeLeafNode(node.preEval, nCorrNaive)
            return

    allIdx = np.arange(data.size) # indices of all data points
    prune(tree, allIdx)
    return