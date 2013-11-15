"""
Dynamics on graphs
"""

import numpy as np


import dynsys, mx
class RandomWalker(dynsys.DiscreteStateSystemBase):
    """
    Random walker class


    >>> import dynpy
    >>> rw = dynpy.graphdynamics.RandomWalker(graph=dynpy.sample_nets.karateclub_net, transCls=dynpy.mx.DenseMatrix )
    >>> rwEnsemble = dynpy.dynsys.DynamicalSystemEnsemble(rw)
    >>> 
    >>> initState = np.zeros(rw.num_vars)
    >>> initState[ 5 ] = 1
    >>> 
    >>> trajectory = rwEnsemble.getTrajectory(initState, last_timepoint=2)[1,1]

    For continuous-time:

    >>> import dynpy
    >>> rw = dynpy.graphdynamics.RandomWalker(graph=dynpy.sample_nets.karateclub_net, discrete_time = False, transCls=dynpy.mx.DenseMatrix )
    >>> rwEnsemble = dynpy.dynsys.DynamicalSystemEnsemble(rw)
    >>> 
    >>> initState = np.zeros(rw.num_vars, 'float')
    >>> initState[ 5 ] = 1
    >>> 
    >>> trajectory = rwEnsemble.getTrajectory(initState, last_timepoint=80, logscale=True)


    """ 
    def __init__(self, graph, discrete_time=True, transCls = None):
        self.cDataType = 'uint8'
        num_vars = graph.shape[0]
        super(RandomWalker, self).__init__(num_vars, discrete_time=discrete_time, transCls=transCls, state_dtypes=self.cDataType)
        graph = np.asarray(graph).astype('double')
        trans = graph / np.atleast_2d( graph.sum(axis = 1) ).T

        if not discrete_time:
            trans = trans - np.eye(*trans.shape)

        self.trans      = self.transCls.finalizeMx( trans )
        self.denseTrans = self.transCls.toDense(self.trans)
        # TODO ELIMINATE NEED FOR SPARSE IF POSSIBLE
        """
        sMap = np.eye(num_vars).astype('int')
        self.ndx2stateDict = {}
        self.state2ndxDict = {}
        for ndx, row in enumerate(sMap):
            state = tuple(row)
            self.ndx2stateDict[ndx] = state
            self.state2ndxDict[state] = ndx

        self.ndx2state = lambda ndx: self.ndx2stateDict[ndx]
        self.state2ndx = lambda stt: self.state2ndxDict[tuple(stt)]
        """
        self.ndx2stateMx = np.eye(num_vars).astype(self.cDataType)
        self.state2ndxDict    = dict(  (mx.hash_np(row),ndx) for ndx, row in enumerate(self.ndx2stateMx) )

    def state2ndx(self, state):
        h = mx.hash_np(state.astype(self.cDataType))
        try:
            return self.state2ndxDict[h]
        except KeyError:
            raise KeyError('%r (hash=%s)' % (state, h))


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
