import numpy as np

class DTnode:
    def __init__(self, attrib, splitPt=None, majority=None):
        self.attrib = attrib
        self.preEval = majority # early evaluation based on training majority
        self.splitPoint = splitPt # if None, then categorical
        self.children = dict()
        self.nCorr = -1 # for storing error info for pruning

    def addChild(self, node, val):
        self.children[val] = node
        
    def isLeaf(self):
        return len(self.children) == 0
        
    def getChild(self, val):
        return self.children[val]
    
    def getValues(self):
        return self.children.keys()
    
    def getChildCorrNum(self):
        return [nd.nCorr for k,nd in self.children.items()]
    
    def makeLeafNode(self, attrib, nCorrPred=-1):
        self.attrib = attrib
        self.preEval = None
        self.children = dict()
        self.nCorr = nCorrPred
    
    def combineChildNodes(self):
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