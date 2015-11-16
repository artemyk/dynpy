"""Module implementing Cellular automaton dynamical systems.
"""

from __future__ import division, print_function, absolute_import
import six
range = six.moves.range
map   = six.moves.map

import numpy as np
from itertools import product as iprod

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
    num_vars : int or list
        The number of cells in the automaton (i.e. the size of the automaton).
        If dim > 1, then this must be list indicating the number of variables
        in each dimension.
    num_neighbors : int
        Number of neighbors in each direction that the update rule depends on.
    rule : int or list
        If mode is 'RULENUMBER', then this should be the update rule, specified 
        as a number representing the truth table of each node.  If mode is 
        'TRUTHTABLE', this should be a list specifying the truthtable.
    mode : {'RULENUMBER','TRUTHTABLE','FUNC'} (default 'RULENUMBER')
        How the rules parameter should be interpreted.
    dim : int (default 1)
        Dimensionality of cellular automata lattice.

    """
    def __init__(self, num_vars, num_neighbors, rule, mode="RULENUMBER", dim=1):

        if dim <= 0:
            raise ValueError('dim must be strictly positive')
        elif dim == 1 and not isinstance(num_vars, list):
            num_vars = [num_vars,]
        
        if not isinstance(num_vars, list) or len(num_vars) != dim:
            raise ValueError('num_vars should be list with %d elements' % dims)

        all_vars = np.array(list(iprod( *[list(range(d)) for d in num_vars] )))
        total_num_vars = len(all_vars)
        all_var_ndxs = { tuple(v):ndx for ndx, v in enumerate(all_vars) }

        neighbor_offsets = np.array(list(iprod(*
            [list(range(-num_neighbors,num_neighbors+1))
             for d in num_vars])))
        if mode == 'FUNC':
            updaterule = self._updatefunc_to_truthtables(len(neighbor_offsets), rule)
        elif mode == "RULENUMBER":
            all_neighbor_cnt = (2*num_neighbors)**dim
            updaterule = list(int2tuple(rule, 2**(all_neighbor_cnt+1)))
        elif mode == "TRUTHTABLE":
            updaterule = rule
        else:
            raise ValueError("Unknown mode %s" % mode)

        rules = []

        for v in all_vars:
            conns = []
            coffsets = v+neighbor_offsets
            conn_address = zip(*[(coffsets[:,d] % num_vars[d]) 
                                 for d in range(dim)])
            conns = [all_var_ndxs[v] for v in conn_address]
            rules.append([v, conns, updaterule])

        super(CellularAutomaton,self).__init__(
            rules=rules, mode='TRUTHTABLES')

