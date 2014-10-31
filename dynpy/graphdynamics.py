"""Module which implements dynamical systems on graph
"""

from __future__ import division, print_function, absolute_import
import six
range = six.moves.range
map   = six.moves.map

import numpy as np

from . import markov, mx, dynsys, utils


class RandomWalker(markov.MarkovChain):
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
        self.cDataType = 'uint8'
        N = graph.shape[0]
        graph = np.asarray(graph).astype('double')

        mxcls = mx.SparseMatrix if issparse else mx.DenseMatrix

        trans = graph / np.atleast_2d( graph.sum(axis=1) ).T

        if not discrete_time:
            trans = trans - np.eye(*trans.shape)

        trans = mxcls.format_mx( trans )

        def iter_states():
            b = np.zeros(N, 'int8')
            for start_state in range(N):
                r = b.copy()
                r[start_state] = 1
                yield mx.hashable_array(r)
        state2ndx_map = utils.readonlydict( (state,ndx) for ndx, state in enumerate(iter_states()) )

        super(RandomWalker, self).__init__(transition_matrix=trans,
            discrete_time=discrete_time, state2ndx_map=state2ndx_map)


