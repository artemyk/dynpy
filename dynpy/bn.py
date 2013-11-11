"""
Boolean Network classes

>>> import dynpy.bn, dynpy.sample_bn_nets, dynpy.dynsys
>>> b = dynpy.bn.BooleanNetworkFromTruthTables(rules=dynpy.sample_bn_nets.yeast)
>>> b.computeStateSpace(dynsys_class=dynpy.dynsys.StochasticDynSys)
>>> atts, attbasins = b.dyn.getAttractorsAndBasins()
>>> print map(len, attbasins)
[1764, 151, 109, 9, 7, 7, 1]
"""
import scipy.sparse as ss
import numpy as np
import dynsys

DEFAULT_DYNAMICS_CLASS = dynsys.SparseStochasticDynSys

def tuple2int(b):
    return int("".join(map(str, map(int, b))), 2)

def int2tuple(i, num_places):
    return tuple(map(int, bin(i)[2:].rjust(num_places, '0')))

class MultivariateDynamicalSystemBase(dynsys.DynamicalSystemBase):
    def __init__(self, num_nodes, node_labels=None, discrete_time = False):
        self.num_nodes       = num_nodes
        self.node_set        = tuple(range(self.num_nodes))
        if node_labels is None:
            node_labels = range(self.num_nodes)
        self.node_labels     = tuple(node_labels) # tuple(map(str, node_labels))
        self.node_label_ndxs = dict((l, ndx) for ndx, l in enumerate(self.node_labels))
        super(MultivariateDynamicalSystem).__init__(discrete_time)

class BooleanNetworkBase(MultivariateDynamicalSystemBase):
    def __init__(self, num_nodes, *kargs, **kwargs):

        super(BooleanNetworkBase, self).__init__(num_nodes, *kargs, **kwargs)

    """
    Prints the attractors and basin of the Boolean Network object


    """
    def printAttractorsAndBasins(self):
        basinAtts, basinStates = self.dyn.getAttractorsAndBasins()
        row_format ="{:>7}" * self.num_nodes
        for cBasinNdx in range(len(basinAtts)):
            print "* BASIN %d : %d States" % (cBasinNdx, len(basinStates[cBasinNdx]))
            print "ATTRACTORS:"
            print row_format.format(*self.node_labels)
            for att in basinAtts[cBasinNdx]:
                print row_format.format(*int2tuple(att, self.num_nodes))
            print
            print


class BooleanNetworkWithTrans(BooleanNetworkBase):
    def checkTransitionMatrix(self):
        expected_shape = 2 ** self.num_nodes
        if self.trans.shape[0] != expected_shape:
            raise Exception(
                "transition matrix shape is %s, but expected first dimension to be 2^%d=%d" %
                (self.trans.shape, num_nodes, expected_shape))
        if self.trans.shape[1] != expected_shape:
            raise Exception(
                "transition matrix shape is %s, but expected second dimension to be 2^%d=%d" %
                (self.trans.shape, num_nodes, expected_shape))
        super(BooleanNetworkWithTrans,self).checkTransitionMatrix()


class BooleanNetworkFromTruthTables(BooleanNetworkBase):
    """
    Rules passed in a list with each element being: [nodename, inputNodes, truthtable ] . 
    For example, [ 'NodeName1',['NodeName2','NodeName4'],[0,0,0,1] ], ]

    """

    def __init__(self, rules, *kargs, **kwargs):
        node_labels = [lbl for (lbl, inputs, table) in rules]
        self.rules = rules
        num_nodes = len(self.rules)
        super(BooleanNetworkFromTruthTables, self).__init__(num_nodes = num_nodes, node_labels = node_labels, rules=rules, **kwargs)

    def getNodeNextState(self, nodeIndex, inputs):
        return self.rules[nodeIndex][2][-1 - tuple2int(inputs)]

    def iterateState(self, startState):
        """
        TODO: Use this to generate fast weave code, or at least make faster
        """
        return tuple(self.getNodeNextState(i, [startState[self.node_label_ndxs[cInput]] for cInput in self.rules[i][1]]) for i in xrange(self.num_nodes) )

    def computeTransitionMatrix(self):
        trans_size = 2 ** self.num_nodes
        # Builds the actual state transition graph

        startStates = [int2tuple(s, self.num_nodes) for s in xrange(trans_size)]
        trans = dynsys_class.getEditableBlankTransMx(num_states = trans_size)
        for s in startStates:
            trans[ tuple2int(s) , tuple2int(self.getNextState(s)) ] = 1.

        self.trans = dynsys_class.finalizeTransMx(trans)
        super(BooleanNetworkFromTruthTables,self).checkTransitionMatrix()

    def getStructuralGraph(self):
        """    
        Get structural connectivity graph for BN
        """
        mx = np.zeros(shape=(self.num_nodes, self.num_nodes))
        for ndx, (node, inputs, table) in enumerate(self.rules):
            ix1 = self.node_label_ndxs[node]
            for i in inputs:
                mx[self.node_label_ndxs[self.rules[i][0]]][ix1] = 1.0


class BooleanNetworkFromFuncs(BooleanNetworkFromTruthTables):
    """
    Get structural connectivity graph for BN
    Rules passed in a list with each element being: [nodename, inputNodes, booleanfunc ] . For example:
    [ 'NodeName1',['NodeName2','NodeNam4'],lambda (NodeName2,NodeName4): NodeName2 and NodeName4 ],

    For example:

    >>> import dynpy.bn, dynpy.dynsys
    >>> b = dynpy.bn.BooleanNetworkFromFuncs(rules=[['A',['A','B'],lambda A,B: A and B],['B',['A','B'],lambda A,B: A or B],])
    >>> b.computeStateSpace( dynsys_class=dynpy.dynsys.StochasticDynSys )
    >>> print b.dyn.trans
    [[ 1.  0.  0.  0.]
     [ 0.  1.  0.  0.]
     [ 0.  1.  0.  0.]
     [ 0.  0.  0.  1.]]

    or, for sparse transition matrices:

    >>> import dynpy.bn, dynpy.dynsys
    >>> b = dynpy.bn.BooleanNetworkFromFuncs(rules=[['A',['A','B'],lambda A,B: A and B],['B',['A','B'],lambda A,B: A or B],])
    >>> b.computeStateSpace( dynsys_class=dynpy.dynsys.SparseStochasticDynSys )
    >>> print b.dyn.trans
      (0, 0)	1.0
      (1, 1)	1.0
      (2, 1)	1.0
      (3, 3)	1.0

    """

    def __init__(self, rules):
        super(BooleanNetworkFromFuncs, self).__init__(rules=rules)

    def getNodeNextState(self, nodeIndex, inputs):
        return self.rules[nodeIndex][2](*inputs)
 


if __name__ == '__main__':
    import sys, os, doctest
    sys.path = [os.path.abspath("..")] + sys.path
    verbose = True 
    r = doctest.testmod(None,None,None,verbose, None, doctest.NORMALIZE_WHITESPACE)
    sys.exit( r[0] )

