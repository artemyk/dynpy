"""Module implementing Boolean Network classes
"""
from __future__ import division, print_function, absolute_import

import sys
if sys.version_info >= (3, 0):
    xrange = range

import inspect
import collections
import itertools

import numpy as np

from . import dynsys
from . import caching


def tuple2int(b):
    """Helper function which converts a binary representation (e.g.,
        ``[1,0,1]``) into an integer
    """
    return int("".join(map(str, map(int, b))), 2)


def int2tuple(i, num_places):
    """Helper function which converts an integer into a binary representation
    (in the form of a numpy array of 0s and 1s). The binary representation will
    be `num_places` long, with extra places padded with 0s.
    """
    return np.array(list(bin(i)[2:].rjust(num_places, '0')), dtype='uint8')

class BooleanNetwork(dynsys.VectorDynamicalSystem):

    """
    A class for Boolean Network dynamical systems.

    The network is specified using the `rules` parameter, which is a list with
    one element for each Boolean variable.  Each of these elements is itself a
    list, containing 3 parts:

        * The name of the current variable
        * The names of the variables that have inputs to the current variable
        * The dynamical update rule for this variable

    The dynamical update rules can be passed in in two ways.  If the `mode`
    parameter is equal to `'TRUTHTABLES'` (the default), then it should be a
    truth table, represented as a list of 0s and 1s. The first element of this
    list corresponds to the desired output when all the inputs are on (i.e., are
    all 1s), while the last element of this list corresopnds to the desired
    output when all the inputs are off.  For example:

    >>> #      output when         1 1 0 0
    >>> #        inputs are:       1 0 1 0
    >>> r = [ ['x1', ['x1','x2'], [1,0,0,0]],
    ...       ['x2', ['x1','x2'], [1,1,1,0]] ]
    >>> import dynpy
    >>> bn1 = dynpy.bn.BooleanNetwork(rules = r)

    The other way to pass in the dynamical update rules is to set the `mode`
    parameter to `'FUNCS'`, and specify the update rule of each variable as a
    Python function that takes in the inputs as arguments:

    >>> r = [ ['x1', ['x1','x2'], lambda x1,x2: (x1 and x2) ],
    ...       ['x2', ['x1','x2'], lambda x1,x2: (x1 or  x2) ] ]
    >>> import dynpy
    >>> bn1 = dynpy.bn.BooleanNetwork(rules = r, mode='FUNCS')

    Parameters
    ----------
    rules : list
        The definition of the Boolean network, as described above
    mode : {'TRUTHTABLES','FUNCS'}, optional
        Specifies how the update functions are defined, 'TRUTHTABLES' is default
    """

    rules = None #: The provided definition of the Boolean network

    def __init__(self, rules, mode='TRUTHTABLES'):

        var_names = [lbl for (lbl, inputs, table) in rules]
        self.rules = rules
        num_vars = len(self.rules)

        super(BooleanNetwork, self).__init__(
            num_vars, var_names, discrete_time=True)

        ALLOWED_MODES = ['TRUTHTABLES', 'FUNCS']
        if mode not in ALLOWED_MODES:
            raise Exception('Parameter mode should be one of %s'%ALLOWED_MODES)

        if mode == 'TRUTHTABLES':
            self.getVarNextState = self._getVarNextStateTT
            for r in self.rules:
                if not isinstance(r[2], collections.Iterable):
                    raise Exception('Truth tables should be specified as ' +
                                    'iterable, not %s' % type(r[2]))

        elif mode == 'FUNCS':
            self.getVarNextState = self._getVarNextStateFuncs
            for r in self.rules:
                if not inspect.isfunction(r[2]):
                    raise Exception(
                        'Boolean functions should be specified as functions')

        self.state2ndx = tuple2int

    @caching.cached_data_prop
    def ndx2stateMx(self):
        """``(num_states, num_vars)``-shaped matrix that maps from state indexes
        to representations in terms of activations of the Boolean variables.
        """
        num_states = 2**self.num_vars
        state_iter = itertools.chain(*self.states())
        ndx2stateMx = np.fromiter(state_iter, dtype='u1')
        return np.reshape(ndx2stateMx, newshape=(num_states, self.num_vars))        

    def _getVarNextStateTT(self, varIndex, inputs):
        """Execute update rule when network is specified using truthtables.
        Repointed in constructor."""
        return self.rules[varIndex][2][-1 - tuple2int(inputs)]

    def _getVarNextStateFuncs(self, varIndex, inputs):
        """Execute update rule when network is specified using functions.
        Repointed in constructor."""
        return self.rules[varIndex][2](*inputs)

    def states(self):
        num_states = 2**self.num_vars
        return (int2tuple(s, self.num_vars) for s in xrange(num_states))

    """
    @caching.cached_data_prop
    def trans(self):
        #: The transition matrix, either as a numpy array (for dense
        #: representations) or scipy.sparse matrix (for sparse representations)
        
        if self.num_vars > 20:
            raise Exception('Computing transition matrix for a %d-variable BN '+
                            'will take too long' % self.num_vars)

        N = 2 ** self.num_vars
        # Builds the actual state transition graph

        trans = self.updateCls.createEditableZerosMx(shape=(N, N))

        for s, curS in enumerate(self.ndx2stateMx):
            trans[s, self.state2ndx(self.iterateOneStep(curS))] = 1.

        trans = self.updateCls.finalizeMx(trans)
        self.checkTransitionMatrix(trans)
        return trans
    """

    @caching.cached_data_prop
    def _inputs(self):
        """Remaps inputs from being specified by variable names to being
        specified by variable indexes. Makes update functions run faster.
        """
        return [
            [self.var_name_ndxs[cInput] for cInput in self.rules[i][1]]
            for i in range(self.num_vars)
        ]

    def printAttractorsAndBasins(self):
        """Prints the attractors and basin of the Boolean Network object

        >>> import dynpy
        >>> rules = [ ['a',['a','b'],[1,1,1,0]],['b',['a','b'],[1,0,0,0]]]
        >>> bn = dynpy.bn.BooleanNetwork(rules=rules)
        >>> bn.printAttractorsAndBasins()
        * BASIN 0 : 2 States
        ATTRACTORS:
              a      b
              1      0
        --------------------------------------------------------------------------------
        * BASIN 1 : 1 States
        ATTRACTORS:
              a      b
              1      1
        --------------------------------------------------------------------------------
        * BASIN 2 : 1 States
        ATTRACTORS:
              a      b
              0      0
        --------------------------------------------------------------------------------

        """
        basinAtts, basinStates = self.getAttractorsAndBasins()
        row_format = "{:>7}" * self.num_vars
        for cBasinNdx in range(len(basinAtts)):
            print("* BASIN %d : %d States" %
                (cBasinNdx, len(basinStates[cBasinNdx])))
            print("ATTRACTORS:")
            print(row_format.format(*self.var_names))
            for att in basinAtts[cBasinNdx]:
                print(row_format.format(*int2tuple(att, self.num_vars)))
            print("".join(['-', ] * 80))

    def checkTransitionMatrix(self, trans):
        """Internally used function that checks the integrity/format of the
        generated transition matrix.
        """
        expected_shape = 2 ** self.num_vars
        if trans.shape[0] != expected_shape:
            raise Exception("transition matrix shape is %s, " +
                "but expected first dimension to be 2^%d=%d" %
                (trans.shape, self.num_vars, expected_shape))
        if trans.shape[1] != expected_shape:
            raise Exception( "transition matrix shape is %s, " +
                "but expected second dimension to be 2^%d=%d" %
                (trans.shape, self.num_vars, expected_shape))
        super(BooleanNetwork, self).checkTransitionMatrix(trans)

    def _iterateOneStepDiscrete(self, startState):
        """Run one interation of Boolean network.  Repointed in parent class
        constructor."""
        return np.array([self.getVarNextState(i, startState[self._inputs[i]])
                         for i in range(self.num_vars)])

    def getStructuralGraph(self):
        """
        Get graph of strutural connectivity

        Returns
        -------
        numpy array
            Adjacency matrix representing which variables have inputs from
            which other variables
        """
        mx = np.zeros(shape=(self.num_vars, self.num_vars))
        for ndx, (var, inputs, table) in enumerate(self.rules):
            ix1 = self.var_name_ndxs[var]
            for i in inputs:
                mx[self.var_name_ndxs[self.rules[i][0]]][ix1] = 1.0

