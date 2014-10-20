from __future__ import division, print_function, absolute_import
import six
range = six.moves.range

import dynpy
import numpy as np

def test_from_deterministic():
	r = [['x1', ['x1','x2'], lambda x1,x2: (x1 and x2) ],
         ['x2', ['x1','x2'], lambda x1,x2: (x1 or  x2) ]]
	bn = dynpy.bn.BooleanNetwork(rules=r, mode='FUNCS')
	mc = dynpy.markov.MarkovChain.from_deterministic_system(bn)

	expected = np.array([[ 1.,  0.,  0.,  0.],
						 [ 0.,  1.,  0.,  0.],
						 [ 0.,  1.,  0.,  0.],
						 [ 0.,  0.,  0.,  1.]])

	assert(np.array_equal(mc.transition_matrix, expected))

def test_sampler():
	rw = dynpy.graphdynamics.RandomWalker(graph=dynpy.sample_nets.karateclub_net)
	sampler = dynpy.markov.MarkovChainSampler(rw)

	cur_state = np.zeros(rw.transition_matrix.shape[0])
	cur_state[ 5 ] = 1
	sampler.iterate(cur_state)
