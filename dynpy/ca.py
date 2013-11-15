"""
Cellular automaton.
"""

import bn, dynsys

class CellularAutomaton(bn.BooleanNetwork):
    """
    Create cellular automaton object

    """
    def __init__(self, num_vars, num_neighbors, ca_rule_number, transCls = None ):
        truth_table = bn.int2tuple(ca_rule_number, 2**(2*num_neighbors+1))
        rules = []
        for i in range(num_vars):
            rules.append([
                i, [(i+n) % num_vars for n in range(-num_neighbors, num_neighbors+1)], truth_table
            ])
        super(CellularAutomaton,self).__init__(rules=rules, transCls = transCls)


if __name__ == '__main__':
    import sys, os, doctest
    import numpy as np
    sys.path = [os.path.abspath("..")] + sys.path
    verbose = True 
    
    num_vars = 100
    num_steps = 50
    ca = CellularAutomaton(num_vars=num_vars, num_neighbors=1, ca_rule_number=110)
    spacetime = np.zeros(shape=(num_steps,num_vars), dtype='int')

    initState = np.zeros(num_vars)
    initState[int(num_vars/2)] = 1.0

    cState = initState.copy()
    for i in range(num_steps):
        spacetime[i,:] = cState
        cState = ca.iterate(cState)

    labels = np.array(['.','#'])
    for line in spacetime:
        print "".join('#' if e == 1.0 else '.' for e in line)


    spacetime = ca.getTrajectory(initState, num_steps)
    for line in spacetime:
        print "".join('#' if e == 1.0 else '.' for e in line)

    #r = doctest.testmod(None,None,None,verbose, None, doctest.NORMALIZE_WHITESPACE)
    #sys.exit( r[0] )
