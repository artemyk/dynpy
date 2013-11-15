"""
Boolean Network classes

"""
import inspect
import collections

import scipy.sparse as ss
import numpy as np

import dynsys, mx, caching


def tuple2int(b):
    return int("".join(map(str, map(int, b))), 2)


def int2tuple(i, num_places):
    return np.array( list(bin(i)[2:].rjust(num_places, '0')) , dtype='uint8' )


class BooleanNetwork(dynsys.DiscreteStateSystemBase):
    """
    Get structural connectivity graph for BN
    Rules passed in a list with each element being: [nodename, inputNodes, booleanfunc ] . For example:
    [ 'NodeName1',['NodeName2','NodeNam4'],lambda (NodeName2,NodeName4): NodeName2 and NodeName4 ],

    For example:

    >>> import dynpy.bn, dynpy.dynsys
    >>> rules = [['A',['A','B'],lambda A,B: A and B],['B',['A','B'],lambda A,B: A or B],]
    >>> b = dynpy.bn.BooleanNetwork(rules=rules, mode = 'FUNCS', transCls=dynpy.mx.DenseMatrix)
    >>> print b.trans
    [[ 1.  0.  0.  0.]
     [ 0.  1.  0.  0.]
     [ 0.  1.  0.  0.]
     [ 0.  0.  0.  1.]]

    or, for sparse transition matrices:

    >>> import dynpy.bn, dynpy.dynsys, dynpy.mx
    >>> rules = [['A',['A','B'],lambda A,B: A and B],['B',['A','B'],lambda A,B: A or B],]
    >>> b = dynpy.bn.BooleanNetwork(rules=rules, mode = 'FUNCS', transCls=dynpy.mx.SparseMatrix)
    >>> print str(b.trans).replace('\\t',' ')
      (0, 0) 1.0
      (1, 1) 1.0
      (2, 1) 1.0
      (3, 3) 1.0

    """
    
    def __init__(self, rules, mode='TRUTHTABLES', transCls=None):
        var_names = [lbl for (lbl, inputs, table) in rules]
        self.rules = rules
        num_vars = len(self.rules)

        super(BooleanNetwork, self).__init__(num_vars     = num_vars,
                                             var_names    = var_names, 
                                             transCls     = transCls,
                                             discrete_time= True,
                                             state_dtypes = 'uint8')

        ALLOWED_MODES = ['TRUTHTABLES', 'FUNCS']
        if mode not in ALLOWED_MODES:
            raise Exception('Parameter mode should be one of %s' % ALLOWED_MODES)

        if mode == 'TRUTHTABLES':
            self.getNodeNextState = self._getNodeNextStateTT
            for r in self.rules:
                if not isinstance(r[2], collections.Iterable):
                    raise Exception('Truth tables should be specified as iterable, not %s' % type(r[2]))
        elif mode == 'FUNCS':
            self.getNodeNextState = self._getNodeNextStateFuncs
            for r in self.rules:
                if not inspect.isfunction(r[2]):
                    raise Exception('Boolean functions should be specified as functions')


        self.state2ndx = tuple2int

    def _getNodeNextStateTT(self, nodeIndex, inputs):
        return self.rules[nodeIndex][2][-1 - tuple2int(inputs)]

    def _getNodeNextStateFuncs(self, nodeIndex, inputs):
        return self.rules[nodeIndex][2](*inputs)

    @caching.cached_data_prop
    def state2ndxMx(self):
        state2ndxMx = np.zeros( (2 ** self.num_vars, self.num_vars) , dtype='u1' )
        for s in range(state2ndxMx.shape[0]):
            state2ndxMx[s,:] = int2tuple(s, self.num_vars)
        return state2ndxMx

    @caching.cached_data_prop
    def trans(self):
        if self.num_vars > 20:
            raise Exception('Computing transition matrix for a %d-node BN will take too long' % self.num_vars)

        trans_size = 2 ** self.num_vars
        # Builds the actual state transition graph

        trans = self.transCls.createEditableZerosMx(shape=(trans_size, trans_size))

        for s, curS in enumerate(self.state2ndxMx):
            trans[s, self.state2ndx( self.iterateOneStep( curS ) )] = 1.

        trans = self.transCls.finalizeMx(trans)
        self.checkTransitionMatrix(trans)
        return trans


    @caching.cached_data_prop
    def _inputs(self):
        return [
                 [ self.var_name_ndxs[cInput] for cInput in self.rules[i][1] ] 
                 for i in range(self.num_vars)
               ]

    """
    Prints the attractors and basin of the Boolean Network object

    """
    def printAttractorsAndBasins(self):
        basinAtts, basinStates = self.dyn.getAttractorsAndBasins()
        row_format = "{:>7}" * self.num_vars
        for cBasinNdx in range(len(basinAtts)):
            print "* BASIN %d : %d States" % (cBasinNdx, len(basinStates[cBasinNdx]))
            print "ATTRACTORS:"
            print row_format.format(*self.var_names)
            for att in basinAtts[cBasinNdx]:
                print row_format.format(*int2tuple(att, self.num_vars))
            print
            print

    def checkTransitionMatrix(self, trans):
        expected_shape = 2 ** self.num_vars
        if trans.shape[0] != expected_shape:
            raise Exception(
                "transition matrix shape is %s, but expected first dimension to be 2^%d=%d" %
                (trans.shape, num_vars, expected_shape))
        if trans.shape[1] != expected_shape:
            raise Exception(
                "transition matrix shape is %s, but expected second dimension to be 2^%d=%d" %
                (trans.shape, num_vars, expected_shape))
        super(BooleanNetwork, self).checkTransitionMatrix(trans)

    def _iterateOneStepDiscrete(self, startState):
        return np.array([self.getNodeNextState(i, startState[self._inputs[i]]) for i in xrange(self.num_vars)])


    def getStructuralGraph(self):
        """    
        Get structural connectivity graph for BN
        """
        mx = np.zeros(shape=(self.num_vars, self.num_vars))
        for ndx, (node, inputs, table) in enumerate(self.rules):
            ix1 = self.var_name_ndxs[node]
            for i in inputs:
                mx[self.var_name_ndxs[self.rules[i][0]]][ix1] = 1.0





if __name__ == '__main__':
    import sys
    import os
    import doctest
    sys.path = [os.path.abspath("..")] + sys.path
    verbose = True
    r = doctest.testmod(None, None, None, verbose, None) # , doctest.NORMALIZE_WHITESPACE)
    sys.exit(r[0])
