from __future__ import division, print_function, absolute_import

from nose.tools import raises

import six
range = six.moves.range

import dynpy
import numpy as np

bnrules = [['x1', ['x1','x2'], lambda x1,x2: (x1 and x2) ],
         ['x2', ['x1','x2'], lambda x1,x2: (x1 or  x2) ]]

def test_from_deterministic():
	bn = dynpy.bn.BooleanNetwork(rules=bnrules, mode='FUNCS')
	mc = dynpy.markov.MarkovChain.from_deterministic_system(bn)

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

def _test_marginalize(issparse):
    bn = dynpy.bn.BooleanNetwork(rules=bnrules, mode='FUNCS')
    bnensemble = dynpy.markov.MarkovChain.from_deterministic_system(bn, issparse=issparse)
    marg = dynpy.markov.MarkovChain.marginalize(bnensemble, [0])

    expected = np.array([[1., 0.], [0.5, 0.5]])

    assert(np.array_equal(dynpy.mx.todense(marg.transition_matrix), expected))

def test_marginalize_dense():
	_test_marginalize(False)

def test_marginalize_sparse():
	_test_marginalize(True)

@raises(ValueError)
def test_check_transition_matrix_not_square():
	dynpy.markov.MarkovChain(transition_matrix=np.zeros(shape=[1,2]))

def test_check_transition_matrix_discrete_sum():
	dynpy.markov.MarkovChain(transition_matrix=np.eye(2))

def test_check_transition_matrix_cont_sum():
	dynpy.markov.MarkovChain(transition_matrix=np.zeros((2,2)),
		discrete_time=False)

@raises(ValueError)
def test_check_transition_matrix_discrete_wrongsum():
	dynpy.markov.MarkovChain(transition_matrix=np.ones((2,2)))

@raises(ValueError)
def test_check_transition_matrix_cont_wrongsum():
	dynpy.markov.MarkovChain(transition_matrix=np.ones((2,2)),
		discrete_time=False)


