"""Module which implements dynamical systems on graph
"""

import numpy as np


import dynsys, mx
class RandomWalker(dynsys.DiscreteStateSystemBase):
    """
    This intializes a stochastic dynamical system representing a random walker on a graph.

    Parameters
    ----------
    graph : numpy array
        Matrix representing the adjacency or weighted connectivity of the underlying graph
    discrete_time : bool, optional
        Whether walker should follow discrete (default) or continuous time dynamics.  Only discrete time
        dynamics are supported for individual walkers, though a distribution of walkers created using the
        :class:`dynpy.dynsys.MarkovChain` supports both.
    transCls : {:class:`dynpy.mx.DenseMatrix`, :class:`dynpy.mx.SparseMatrix`}, optional 
        Wether to use sparse or dense matrices for the transition matrix.  Default set by `dynpy.dynsys.DEFAULT_TRANSMX_CLASS`

    """

    #: Transition matrix of random walker system
    trans = None   

    #: ``(num_states, num_vars)``-shaped matrix which maps from integer state indexes to their representations
    #: in terms of the values of the system variables. 
    ndx2stateMx  = None

    def __init__(self, graph, discrete_time=True, transCls = None):
        self.cDataType = 'uint8'
        num_vars = graph.shape[0]
        super(RandomWalker, self).__init__(num_vars, discrete_time=discrete_time, transCls=transCls, state_dtypes=self.cDataType)
        graph = np.asarray(graph).astype('double')
        trans = graph / np.atleast_2d( graph.sum(axis = 1) ).T

        if not discrete_time:
            trans = trans - np.eye(*trans.shape)

        self.trans      = self.transCls.finalizeMx( trans )
        self.checkTransitionMatrix(self.trans)
        self.denseTrans = self.transCls.toDense(self.trans)        
        self.ndx2stateMx = np.eye(num_vars).astype(self.cDataType)

    def _iterateOneStepDiscrete(self, startState):
        return self.ndx2stateMx[np.random.choice( self.num_vars, None, replace=True, p=np.ravel( self.denseTrans[self.state2ndx(startState),:])), :]

    def _iterateContinuous(self, startState, max_time = 1.0):
        raise NotImplementedError

if __name__ == '__main__':
    import sys
    import os
    import doctest
    sys.path = [os.path.abspath("..")] + sys.path
    verbose = True
    r = doctest.testmod(None, None, None, verbose, None) # , doctest.NORMALIZE_WHITESPACE)
    sys.exit(r[0])
