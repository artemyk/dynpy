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
    >>> ca = CellularAutomaton(num_vars=50, num_neighbors=1, rule=110)
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
    rule : int or list
        If mode is 'RULENUMBER', then this should be the update rule, specified 
        as a number representing the truth table of each node.  If mode is 
        'TRUTHTABLE', this should be a list specifying the truthtable.
    mode : {'RULENUMBER','TRUTHTABLE'} (default 'RULENUMBER')
        How the rules parameter should be interpreted. 

    """
    def __init__(self, num_vars, num_neighbors, rule, mode="RULENUMBER"):
        if mode == "RULENUMBER":
            truth_table = list(int2tuple(rule, 2**(2*num_neighbors+1)))
        elif mode == "TRUTHTABLE":
            truth_table = rule
        else:
            raise ValueError("Unknown mode %s" % mode)

        rules = []
        for i in range(num_vars):
            conns = [(i+n) % num_vars
                     for n in range(-num_neighbors, num_neighbors+1)]
            rules.append([i, conns, truth_table])
        super(CellularAutomaton,self).__init__(rules=rules)

