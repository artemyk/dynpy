"""
Cellular automaton.  For example:

>>> from pylab import *
>>> from matplotlib.patches import Ellipse
>>> delta = 45.0 # degrees
>>> angles = arange(0, 360+delta, delta)
>>> ells = [Ellipse((1, 1), 4, 2, a) for a in angles]
>>> a = subplot(111, aspect='equal')
>>> for e in ells:
>>>     e.set_clip_box(a.bbox)
>>>     e.set_alpha(0.1)
>>>     a.add_artist(e)
>>> 
>>> xlim(-2, 4)
>>> ylim(-1, 3)
>>> 
>>> show()

and then, what do you know...

.. plot:: pyplots/ellipses.py
   :include-source:

"""

import bn
class CellularAutomaton(bn.BooleanNetworkFromTruthTables):
    """
    Create cellular automaton object

    >>>ca = CellularAutomaton(10)


    """
    def __init__(self, num_nodes, num_neighbors, ca_rule_number):
        truth_table = bn.int2tuple(ca_rule_number, 2**(2*num_neighbors+1))
        rules = []
        for i in range(num_nodes):
            rules.append([
                i, [(i+n) % num_nodes for n in range(-num_neighbors, num_neighbors+1)], truth_table
            ])
        super(CellularAutomaton,self).__init__(rules=rules)


if __name__ == '__main__':
    import sys, os, doctest
    import numpy as np
    sys.path = [os.path.abspath("..")] + sys.path
    verbose = True 
    
    num_nodes = 100
    num_steps = 50
    ca = CellularAutomaton(num_nodes=num_nodes, num_neighbors=1, ca_rule_number=110)
    spacetime = np.zeros(shape=(num_steps,num_nodes), dtype='int')

    cState = np.zeros(num_nodes)
    cState[int(num_nodes/2)] = 1.0

    for i in range(num_steps):
        spacetime[i,:] = cState
        cState = ca.getNextState(cState)

    labels = np.array(['.','#'])
    print labels[spacetime.astype('int')]
    for line in spacetime:
        print "".join('#' if e == 1.0 else '.' for e in line)
    #r = doctest.testmod(None,None,None,verbose, None, doctest.NORMALIZE_WHITESPACE)
    #sys.exit( r[0] )
