from __future__ import division, print_function, absolute_import

from nose.tools import raises
from numpy.testing import assert_array_equal, assert_allclose

import six
range = six.moves.range

import dynpy
from dynpy.markov import MarkovChain
from dynpy.bn     import BooleanNetwork

import numpy as np

bnrules = [['x1', ['x1','x2'], lambda x1,x2: (x1 and x2) ],
         ['x2', ['x1','x2'], lambda x1,x2: (x1 or  x2) ]]

def test_from_deterministic():
	bn = BooleanNetwork(rules=bnrules, mode='FUNCS')
	mc = MarkovChain.from_deterministic_system(bn)

	expected = np.array([[ 1.,  0.,  0.,  0.],
						 [ 0.,  1.,  0.,  0.],
						 [ 0.,  1.,  0.,  0.],
						 [ 0.,  0.,  0.,  1.]])

	assert_array_equal(dynpy.mx.todense(mc.transition_matrix), expected)

def test_sampler():
	rw = dynpy.graphdynamics.RandomWalker(graph=dynpy.sample_nets.karateclub_net)
	sampler = dynpy.markov.MarkovChainSampler(rw)

	cur_state = np.zeros(rw.transition_matrix.shape[0])
	cur_state[ 5 ] = 1
	sampler.iterate(cur_state)

def _test_marginalize(issparse, initial_dist=None, expected=[[1., 0.], [0.5, 0.5]]):
    bn = BooleanNetwork(rules=bnrules, mode='FUNCS')
    bnensemble = MarkovChain.from_deterministic_system(
    				bn, issparse=issparse)
    marg = MarkovChain.marginalize(bnensemble, [0], initial_dist=initial_dist)
    trans = dynpy.mx.todense(marg.transition_matrix)
    assert_array_equal(trans, np.core.asarray(expected))

def test_marginalize_dense():
	_test_marginalize(False, expected=[[1., 0.], [0.5, 0.5]])

def test_marginalize_sparse():
	_test_marginalize(True , expected=[[1., 0.], [0.5, 0.5]])

def test_marginalize_initial_withnan():
	dist = np.ravel(np.array([0,0,0,1]))
	_test_marginalize(False, initial_dist=dist, 
		expected=[[np.nan, np.nan], [0, 1]])

def test_marginalize_initial():
	dist = np.ravel(np.array([0.1,0.1,0.1,0.7]))
	_test_marginalize(False, initial_dist=dist, 
		expected=[[ 1., 0.], [0.125, 0.875]])

@raises(ValueError)
def test_check_transition_matrix_not_square():
	MarkovChain(transition_matrix=np.zeros(shape=[1,2]))

def test_check_transition_matrix_discrete_sum():
	MarkovChain(transition_matrix=np.eye(2))

def test_check_transition_matrix_cont_sum():
	MarkovChain(transition_matrix=np.zeros((2,2)),
		discrete_time=False)

@raises(ValueError)
def test_check_transition_matrix_discrete_wrongsum():
	MarkovChain(transition_matrix=np.ones((2,2)))

@raises(ValueError)
def test_check_transition_matrix_cont_wrongsum():
	MarkovChain(transition_matrix=np.ones((2,2)),
		discrete_time=False)

def _test_attractors(issparse):
	bn = BooleanNetwork(rules=dynpy.sample_nets.yeast_cellcycle_bn)
	bn_ensemble = MarkovChain.from_deterministic_system(bn, issparse=issparse)

	final_dist = bn_ensemble.iterate(bn_ensemble.get_uniform_distribution(), 100)
	bn.get_attractor_basins()
	total_states = 2 ** bn.num_vars

	dist2 = np.zeros(bn_ensemble.get_uniform_distribution().shape)
	for att, basin in zip(*bn.get_attractor_basins()):
	    basin_size = len(basin)
	    weight = float(basin_size) / total_states
	    dist2[ bn_ensemble.state2ndx(att[0]) ] = weight
	    
	assert_allclose(dist2, np.ravel(dynpy.mx.todense(final_dist)))

def test_attractors_sparse():
	return _test_attractors(True)

def test_attractors_dense():
	return _test_attractors(False)
