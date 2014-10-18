"""Module implementing Cellular automaton dynamical system
"""
from __future__ import division, print_function, absolute_import
import sys
if sys.version_info >= (3, 0):
    xrange = range

from . import bn

class CellularAutomaton(bn.BooleanNetwork):
    """Cellular automaton object.  Constructs an underlying
    :class:`dynpy.bn.BooleanNetwork` on a lattice and with a homogenous update
    function.  Implements periodic boundary conditions.

    For example:

    >>> from dynpy.ca import CellularAutomaton
    >>> import numpy as np
    >>> ca = CellularAutomaton(num_vars=50, num_neighbors=1, ca_rule_number=110)
    >>> initState = np.zeros(ca.num_vars)
    >>> initState[int(ca.num_vars/2)] = 1.0
    >>> for line in ca.getTrajectory(initState, 10):
    ...   print("".join('#' if e == 1.0 else '-' for e in line))
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
        truth_table = bn.int2tuple(ca_rule_number, 2**(2*num_neighbors+1))
        rules = []
        for i in range(num_vars):
            conns = [(i+n) % num_vars
                     for n in xrange(-num_neighbors, num_neighbors+1)]
            rules.append([i, conns, truth_table])
        super(CellularAutomaton,self).__init__(rules=rules)

