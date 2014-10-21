"""Module implementing Boolean Network classes
"""
from __future__ import division, print_function, absolute_import

import six
range = six.moves.range

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
    return dynsys.hashable_state(
        np.array(list(map(int,bin(i)[2:].rjust(num_places, '0')))))

class BooleanNetwork(dynsys.DiscreteStateDynamicalSystem, 
    dynsys.VectorDynamicalSystem,
    dynsys.DeterministicDynamicalSystem):

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
    >>> bn1 = dynpy.bn.BooleanNetwork(rules=r, mode='FUNCS')

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
            self._get_var_next_state = self._get_var_next_state_tt
            for r in self.rules:
                if not isinstance(r[2], collections.Iterable):
                    raise Exception('Truth tables should be specified as ' +
                                    'iterable, not %s' % type(r[2]))

        elif mode == 'FUNCS':
            self._get_var_next_state = self._get_var_next_state_funcs
            for r in self.rules:
                if not inspect.isfunction(r[2]):
                    raise Exception(
                        'Boolean functions should be specified as functions')

        self.state2ndx = tuple2int

    def _get_var_next_state_tt(self, varIndex, inputs):
        """Execute update rule when network is specified using truthtables.
        Repointed in constructor."""
        return self.rules[varIndex][2][-1 - tuple2int(inputs)]

    def _get_var_next_state_funcs(self, varIndex, inputs):
        """Execute update rule when network is specified using functions.
        Repointed in constructor."""
        return self.rules[varIndex][2](*inputs)

    def states(self):
        """Returns list of all possible states occupied by system.
        """
        num_states = 2**self.num_vars
        return (int2tuple(s, self.num_vars) for s in range(num_states))

    @caching.cached_data_prop
    def _inputs(self):
        """Remaps inputs from being specified by variable names to being
        specified by variable indexes. Makes update functions run faster.
        """
        return [
            [self.var_name_ndxs[cInput] for cInput in self.rules[i][1]]
            for i in range(self.num_vars)
        ]

    def _iterate_1step_discrete(self, start_state):
        """Run one interation of Boolean network.  iterate is pointed to this
        in parent class constructor."""
        return dynsys.hashable_state(np.array(
            [self._get_var_next_state(v, start_state[self._inputs[v]])
             for v in range(self.num_vars)]))

    def get_structural_graph(self):
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
                mx[self.var_name_ndxs[i],ix1] = 1.0
        return mx
