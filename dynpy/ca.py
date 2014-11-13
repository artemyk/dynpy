"""Module implementing Cellular automaton dynamical systems.
"""

from __future__ import division, print_function, absolute_import
import six
range = six.moves.range
map   = six.moves.map

from . import bn
from .cutils import int2tuple

class CellularAutomaton(bn.BooleanNetwork):
    """Cellular automaton object.  Constructs an underlying
    :class:`dynpy.bn.BooleanNetwork` on a lattice and with a homogenous update
    function.  Implements periodic boundary conditions.

    For example:

    >>> from dynpy.ca import CellularAutomaton
    >>> import numpy as np
    >>> ca = CellularAutomaton(num_vars=50, num_neighbors=1, ca_rule_number=110)
    >>> init_state = np.zeros(ca.num_vars, dtype='uint8')
    >>> init_state[int(ca.num_vars/2)] = 1
    >>> for line in ca.get_trajectory(init_state, 10):
    ...   print("".join('#' if e == 1 else '-' for e in line))
    -------------------------#------------------------
    ------------------------##------------------------
    -----------------------###------------------------
    ----------------------##-#------------------------
    ---------------------#####------------------------
    --------------------##---#------------------------
    -------------------###--##------------------------
    ------------------##-#-###------------------------
    -----------------#######-#------------------------
    ----------------##-----###------------------------

    Parameters
    ----------
    num_vars : int
        The number of cells in the automaton (i.e. the size of the automaton)
    num_neighbors : int
        Number of neighbors that the update rule depends on
    ca_rule_number : int
        The update rule, specified as a number representing the truth table of
        each node

    """
    def __init__(self, num_vars, num_neighbors, ca_rule_number):
        truth_table = list(int2tuple(ca_rule_number, 2**(2*num_neighbors+1)))
        rules = []
        for i in range(num_vars):
            conns = [(i+n) % num_vars
                     for n in range(-num_neighbors, num_neighbors+1)]
            rules.append([i, conns, truth_table])
        super(CellularAutomaton,self).__init__(rules=rules)

