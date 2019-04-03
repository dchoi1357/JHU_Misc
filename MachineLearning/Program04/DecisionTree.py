import numpy as np

class DTnode:
    ''' Decision Tree node class.
    The class implements a node in the decision tree used for classificaiton.
    A tree can be represented as a root node, as well as any sub-trees. 
    '''
    def __init__(self, attrib, splitPt=None, majority=None):
        ''' Constructor for decision tree node.
        Receives three parameters
        - [attrib] is the attribute used to split the data set if it is an
            inner node. If it's a leaf node, it is the classification label
        - [splitPt] is the value for split if the feature is numeric. If the
            feature is categorical, this is None.
        - [majority] is the most common label of all data up to this point of
            the tree. For leaf node, this is None as it would be same as attrib
        '''
        self.attrib = attrib # attribute or classification
        self.preEval = majority # early evaluation based on training majority
        self.splitPoint = splitPt # if None, then categorical
        self.children = dict() # children as dict
        self.nCorr = -1 # for storing error info for pruning

    def addChild(self, node, val):
        ''' Add children to DTnode.
        Children are stored in a dict, where the key is either the value of
        the feature if it's categorical, or < or >= for numeric.
        '''
        self.children[val] = node
        
    def isLeaf(self):
        ''' If the node is a leaf node.
        A leaf node has no children, which means children dict is empty.
        '''
        return len(self.children) == 0
        
    def getChild(self, val):
        ''' Get child node associated with a key.
        '''
        return self.children[val]
    
    def getValues(self):
        ''' Get all values for current node. Returns all possible values of
        features seen at this point.
        '''
        return self.children.keys()
    
    def getChildCorrNum(self):
        ''' Get the number of correctly predicted sample for all child nodes.
        This is used for pruning, as to show whether sub-trees make better 
        prediction on the pruning set than just the majority label.
        '''
        return [nd.nCorr for k,nd in self.children.items()]
    
    def makeLeafNode(self, attrib, nCorrPred=-1):
        ''' Make the current DTnode a leaf node.
        
        Accomplished by removing all child node references, remove majority 
        labels, and setting the attrib to the same as parameter.
        '''
        self.attrib = attrib
        self.preEval = None
        self.children = dict()
        self.nCorr = nCorrPred
    
    def combineChildNodes(self):
        ''' Combine child nodes of trees for which the predictions labels are
        homogenous. Since it does not matter what features there are, the child
        nodes can be removed for simplicity sake. This is performed recursively
        in a depth-first manner, all the way up to the current node.
        '''
        if self.isLeaf():
            return set([self.attrib])
        subLabl = set()
        for k,child in self.children.items(): # loop over all child nodes
            subLabl.update( child.combineChildNodes() )
        if len(subLabl) == 1: # only one class for all child nodes
            self.makeLeafNode( next(iter(subLabl)) )
        return subLabl

    def __repre__(self):
        if self.isLeaf():
            childTxt = 'terminal'
        else:
            childTxt = 'child: ' + str(list(self.children.keys()))
        return '[Node for %s, %s ]'%(self.attrib, childTxt)
    
    def toStr(self, level=0):
        ''' Print string representation of the decision tree rooted at current
        node.

        This is accomplished recursively, where the attribute, value, and 
        classificaion of the leaf node are passed onto the parent node, and
        each sub node has one more level of indent than the parent.
        '''
        if self.isLeaf():
            return 'class: %s\n' % self.attrib
        else:
            ret = 'Attribute [' + self.attrib + "]:\n"
            nx = level + 1
            for key in self.children:
                if self.splitPoint is None: # categorical var
                    txt = '= %s'%key
                else: # numerical var
                    txt = '%s %f'%(key,self.splitPoint)
                ret += " "*nx*4 + 'value %s, '%txt \
                    + self.children[key].toStr(nx)
            return ret
    
    def __str__(self):
        return self.toStr()