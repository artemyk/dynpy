"""
Boolean Network classes

"""
import inspect
import collections

import scipy.sparse as ss
import numpy as np
import dynsys
import mx




def tuple2int(b):
    return int("".join(map(str, map(int, b))), 2)


def int2tuple(i, num_places):
    return tuple(map(int, bin(i)[2:].rjust(num_places, '0')))


class BooleanNetworkBase(dynsys.MultivariateDynamicalSystemBase):

    def __init__(self, num_nodes, node_labels, transMatrixClass=dynsys.DEFAULT_TRANSMX_CLASS):
        super(BooleanNetworkBase, self).__init__(num_nodes=num_nodes,
                                                 node_labels=node_labels, 
                                                 transMatrixClass=transMatrixClass)

    def __getattr__(self, name):
        if name == 'trans':
            self.computeTransitionMatrix()
            return self.trans
        else:
            # Default behaviour
            raise AttributeError(name)

    """
    Prints the attractors and basin of the Boolean Network object

    """
    def printAttractorsAndBasins(self):
        basinAtts, basinStates = self.dyn.getAttractorsAndBasins()
        row_format = "{:>7}" * self.num_nodes
        for cBasinNdx in range(len(basinAtts)):
            print "* BASIN %d : %d States" % (cBasinNdx, len(basinStates[cBasinNdx]))
            print "ATTRACTORS:"
            print row_format.format(*self.node_labels)
            for att in basinAtts[cBasinNdx]:
                print row_format.format(*int2tuple(att, self.num_nodes))
            print
            print

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
        super(BooleanNetworkBase, self).checkTransitionMatrix()



class BooleanNetworkFromTruthTables(BooleanNetworkBase):

    """
    Rules passed in a list with each element being: [nodename, inputNodes, truthtable ] . 
    For example, [ 'NodeName1',['NodeName2','NodeName4'],[0,0,0,1] ], ]

    """

    def __init__(self, rules, transMatrixClass=dynsys.DEFAULT_TRANSMX_CLASS):
        node_labels = [lbl for (lbl, inputs, table) in rules]
        self.rules = rules
        num_nodes = len(self.rules)

        self.checkRules()
        super(BooleanNetworkFromTruthTables, self).__init__(num_nodes=num_nodes, node_labels=node_labels, transMatrixClass=transMatrixClass)


    def checkRules(self):
        for r in self.rules:
            if not isinstance(r[2], collections.Iterable):
                raise Exception('Truth tables should be specified as iterable, not %s' % type(r[2]))

    def getNodeNextState(self, nodeIndex, inputs):
        return self.rules[nodeIndex][2][-1 - tuple2int(inputs)]

    def iterateState(self, startState):
        """
        TODO: Use this to generate fast weave code, or at least make faster
        """
        return tuple(self.getNodeNextState(i, [startState[self.node_label_ndxs[cInput]] for cInput in self.rules[i][1]]) for i in xrange(self.num_nodes))

    def computeTransitionMatrix(self):
        if self.num_nodes > 20:
            raise Exception('Computing transition matrix for a %d-node BN will take too long' % self.num_nodes)

        trans_size = 2 ** self.num_nodes
        # Builds the actual state transition graph

        trans = self.transCls.createEditableZerosMx(shape=(trans_size, trans_size))
        self.ndx2state = lambda (ndx): int2tuple(ndx, self.num_nodes)
        self.state2ndx = tuple2int

        for s in xrange(trans_size):
            trans[s, self.state2ndx( self.iterateState( self.ndx2state(s) ) )] = 1.

        self.trans = self.transCls.finalizeMx(trans)
        self.checkTransitionMatrix()

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
    >>> b = dynpy.bn.BooleanNetworkFromFuncs(rules=[['A',['A','B'],lambda A,B: A and B],['B',['A','B'],lambda A,B: A or B],], transMatrixClass=dynpy.mx.DenseMatrix)
    >>> print b.trans
    [[ 1.  0.  0.  0.]
     [ 0.  1.  0.  0.]
     [ 0.  1.  0.  0.]
     [ 0.  0.  0.  1.]]

    or, for sparse transition matrices:

    >>> import dynpy.bn, dynpy.dynsys, dynpy.mx
    >>> b = dynpy.bn.BooleanNetworkFromFuncs(rules=[['A',['A','B'],lambda A,B: A and B],['B',['A','B'],lambda A,B: A or B],], transMatrixClass=dynpy.mx.SparseMatrix)
    >>> print str(b.trans).replace('\\t',' ')
      (0, 0) 1.0
      (1, 1) 1.0
      (2, 1) 1.0
      (3, 3) 1.0
    
    """

    def checkRules(self):
        for r in self.rules:
            if not inspect.isfunction(r[2]):
                raise Exception('Boolean functions should be specified as functions')

    def getNodeNextState(self, nodeIndex, inputs):
        return self.rules[nodeIndex][2](*inputs)


if __name__ == '__main__':
    import sys
    import os
    import doctest
    sys.path = [os.path.abspath("..")] + sys.path
    verbose = True
    r = doctest.testmod(None, None, None, verbose, None) # , doctest.NORMALIZE_WHITESPACE)
    sys.exit(r[0])
