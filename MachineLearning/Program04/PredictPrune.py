import numpy as np
#from DecisionTree import DTnode

def PredictDTree(tree, data):
    def classify(node, idx):
        ndData = data[idx] # portion of data for this node
        if node.isLeaf():
            result[idx] = node.attrib
            return
        else:
            feat = node.attrib
            if node.splitPoint is None: # categorical
                for k in node.getValues():
                    subIdx = idx[ ndData[feat]==k ]
                    classify(node.getChild(k), subIdx)
            else: # numeric
                lessThan = ndData[feat]<node.splitPoint
                classify(node.getChild('<'), idx[lessThan])
                classify(node.getChild('>='), idx[~lessThan])
            return

    allIdx = np.arange(data.size)
    result = np.empty(data.size, object)
    classify(tree, allIdx)
    return result
  
def PruneDTree(tree, data, actuals):
    def prune(node, idx):
        ndData,ndActs = data[idx],actuals[idx] # data and labels for the node
        if node.isLeaf():
            node.nCorr = sum(node.attrib==ndActs) # save nCorrect preds
            return
        else:
            feat = node.attrib
            if node.splitPoint is None: # categorical
                for k in node.getValues():
                    subIdx = idx[ ndData[feat]==k ]
                    prune(node.getChild(k), subIdx)
            else: # numeric
                lessThan = ndData[feat]<node.splitPoint
                prune(node.getChild('<'), idx[lessThan])
                prune(node.getChild('>='), idx[~lessThan])
            
            node.nCorr = sum(node.getChildCorrNum()) # sum childNodes error nums
            nCorrNaive = sum(ndActs==node.preEval) # nCorr using label majority
            if nCorrNaive > node.nCorr: # if majority class is better than C4.5
                node.makeLeafNode(node.preEval, nCorrNaive)
            return

    allIdx = np.arange(data.size)
    prune(tree, allIdx)
    return