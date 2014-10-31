"""Module which implements dynamical systems on graph
"""

from __future__ import division, print_function, absolute_import
import six
range = six.moves.range
map   = six.moves.map

import numpy as np

from . import markov, mx, dynsys, utils

class RandomWalkerBase(dynsys.DiscreteStateVectorDynamicalSystem,
    dynsys.StochasticDynamicalSystem):
    #TODO: Document

    def __init__(self, graph):
        self.graph = graph
        N = graph.shape[0]
        super(RandomWalkerBase,self).__init__(num_vars=N)

    def states(self):
        for i in range(self.num_vars):
            c = np.zeros(self.num_vars, dtype='int8')
            c[i] = 1
            yield c

    def iterate(self):
        # Should use a sampler from RandomWalkerEnsemble instead
        raise NotImplementedError

class RandomWalkerEnsemble(markov.MarkovChain):
    """This intializes a stochastic dynamical system representing a random
    walker on a graph.dynpy.sample_nets.karateclub_net

    Parameters
    ----------
    graph : numpy array
        Matrix representing the adjacency or weighted connectivity of the
        underlying graph
    discrete_time : bool, optional
        Whether walker should follow discrete (default) or continuous time
        dynamics.  Only discrete time dynamics are supported for individual
        walkers, though a distribution of walkers created using the
        :class:`dynpy.dynsys.MarkovChain` supports both.
    issparse : bool, optional
        Whether to use a sparse or dense transition matrix.

    """

    def __init__(self, graph, discrete_time=True, issparse=True):
        base_sys = RandomWalkerBase(graph)

        mxcls = mx.SparseMatrix if issparse else mx.DenseMatrix

        trans = np.asarray(graph).astype('double') 
        trans = trans / np.atleast_2d( graph.sum(axis=1) ).T

        if not discrete_time:
            trans = trans - np.eye(*trans.shape)

        trans = mxcls.format_mx( trans )

        super(RandomWalkerEnsemble, self).__init__(
            transition_matrix=trans,
            discrete_time=discrete_time, 
            base_sys=base_sys)


