"""Module which implements dynamical systems on graph
"""
from __future__ import division, print_function, absolute_import
import six
range = six.moves.range

import numpy as np

from . import markov, mx, dynsys


class RandomWalker(markov.MarkovChain):
    """This intializes a stochastic dynamical system representing a random
    walker on a graph.

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
    TODO issparse

    """

    def __init__(self, graph, discrete_time=True, issparse=False):
        self.cDataType = 'uint8'
        self.num_vars = graph.shape[0]
        graph = np.asarray(graph).astype('double')

        cls = mx.SparseMatrix if issparse else mx.DenseMatrix

        trans = graph / np.atleast_2d( graph.sum(axis=1) ).T

        self.checkTransitionMatrix(trans)

        if not discrete_time:
            trans = trans - np.eye(*trans.shape)

        trans = cls.formatMx( trans )

        super(RandomWalker, self).__init__(updateOperator=trans,
            discrete_time=discrete_time)

