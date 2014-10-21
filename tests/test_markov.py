from __future__ import division, print_function, absolute_import

from nose.tools import raises

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

	assert(np.array_equal(dynpy.mx.todense(mc.transition_matrix), expected))

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
    assert(dynpy.mx.DenseMatrix.array_equal(trans, np.array(expected)))

def test_marginalize_dense():
	_test_marginalize(False, expected=[[1., 0.], [0.5, 0.5]])

def test_marginalize_sparse():
	_test_marginalize(True , expected=[[1., 0.], [0.5, 0.5]])

def test_marginalize_initial():
	dist = np.ravel(np.array([0,0,0,1]))
	_test_marginalize(False, initial_dist=dist, 
		expected=[[np.nan, np.nan], [0, 1]])


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


