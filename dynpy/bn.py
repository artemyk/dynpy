"""Module implementing Boolean Network classes
"""
from __future__ import division, print_function, absolute_import

import six
range = six.moves.range
map   = six.moves.map

import inspect
import collections

import numpy as np

from . import dynsys

from .cutils import int2tuple
from .bniterate import iterate_1step_truthtable

class BooleanNetwork(dynsys.DiscreteStateVectorDynamicalSystem, 
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
        Specifies how the update functions are defined (as truthtables or as
        Python functions). By default tries to guess
    convert_to_truthtable : bool, optional
        If update functions are passed in as Python, whether to convert them
        to truthtables.  This has initial performance penalty but then leads
        to faster iterations.
    """

    rules = None #: The provided definition of the Boolean network

    def __init__(self, rules, mode=None, convert_to_truthtable=True):

        var_names = [lbl for (lbl, _, _) in rules]
        num_vars = len(rules)

        # Remap inputs from being specified by variable names to being
        # specified by variable indexes. Makes update functions run faster.
        # TODO: PERF: Only do if necessary
        new_rules = []
        var_name_ndxs = { name:ndx for ndx, (name, _, _) in enumerate(rules) }
        for var, input_vars, update in rules:
            if len(input_vars) and isinstance(input_vars[0], str):
                input_vars = [var_name_ndxs[v] for v in input_vars]
            new_rules.append([var, input_vars, update])

        self.rules = new_rules

        if mode is None:
            if hasattr(self.rules[0][2], '__call__'):
                # function update rule is callable, assume it is a function
                mode = 'FUNCS'
            else:
                mode = 'TRUTHTABLES'

        if mode == 'TRUTHTABLES':
            for r in self.rules:
                if not isinstance(r[2], list):
                    raise ValueError('Truth tables should be specified as ' +
                                    'list, not %s' % type(r[2]))
            self._init_truthtables()

        elif mode == 'FUNCS':
            for r in self.rules:
                if not inspect.isfunction(r[2]):
                    raise ValueError(
                        'Boolean functions should be specified as functions')

            if convert_to_truthtable:
                self.rules = [(names, inputs, self._updatefunc_to_truthtables(len(inputs), f))
                              for names, inputs, f in self.rules]
                self._init_truthtables()

            else:
                self._iterate_1step_discrete = self._iterate_1step_discrete_funcs

        else:
            raise ValueError('Invalid mode parameter %s' % mode)

        super(BooleanNetwork, self).__init__(
            num_vars, var_names, discrete_time=True)

    @classmethod
    def _updatefunc_to_truthtables(cls, K, func):
        # Convert Python functions to truth tables
        # TODO: Test
        # TODO: Cythonize
        return [func(*int2tuple(inputstate, K))
                for inputstate in range(2**K - 1, -1, -1)]

    def _init_truthtables(self):
        self._iterate_1step_discrete = six.create_bound_method(iterate_1step_truthtable, self)

    def _iterate_1step_discrete_funcs(self, start_state):
        """Run one interation of Boolean network.  iterate is pointed to this
        in parent class constructor."""

        return np.array(
             [self._get_var_next_state_funcs(v, [start_state[i] for i in self.rules[v][1]])
             for v in range(self.num_vars)], dtype=start_state.dtype)

    def _get_var_next_state_funcs(self, var_index, inputs):
        #: Execute update rule when network is specified using functions.
        #: Repointed in constructor.
        return self.rules[var_index][2](*inputs)

    def states(self):
        """Returns list of all possible states occupied by system.
        """
        num_states = 2**self.num_vars
        return (int2tuple(s, self.num_vars) for s in range(num_states))

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
        for var, inputs, _ in self.rules:
            ix1 = self.var_name_ndxs[var]
            for i in inputs:
                mx[i,ix1] = 1.0
        return mx

    """
    def jacobian(self, state, time=1.0):
        next_state = self.iterate(state, time)
        indexes = np.arange(self.num_vars, dtype='int')

        r = []
        for i in range(self.num_vars):
            perturbed_state = state.copy()
            perturbed_state[i] = 1-perturbed_state[i]
            r.append(np.logical_xor(next_state, self.iterate(perturbed_state)))

        return np.array(r)
    """

