from __future__ import division, print_function, absolute_import

import dynpy
import numpy as np

def test_def_bns():
	r = [
		['x1', ['x1','x2'], [1,0,0,0]],
		['x2', ['x1','x2'], [1,1,1,0]],
	]
	bn1base = dynpy.bn.BooleanNetwork(rules=r)
	bn1 = dynpy.markov.MarkovChain.from_deterministic_system(bn1base)

	r2 = [
		['x1', ['x1','x2'], lambda x1,x2: (x1 and x2) ],
		['x2', ['x1','x2'], lambda x1,x2: (x1 or  x2) ],
	]
	bn2base = dynpy.bn.BooleanNetwork(rules=r2, mode='FUNCS')
	bn2 = dynpy.markov.MarkovChain.from_deterministic_system(bn2base)

	assert((bn1.updateOperator-bn2.updateOperator).max() == 0.0 )
	assert((bn1.updateOperator-bn2.updateOperator).min() == 0.0 )

#def test_ndx2state_mx():
#	bn2 = dynpy.bn.BooleanNetwork(dynpy.sample_nets.yeast_cellcycle_bn)
#	assert(all(bn2.ndx2stateMx[0,:] == np.zeros(bn2.num_vars)))
#	assert(all(bn2.ndx2stateMx[-1,:] == np.ones(bn2.num_vars)))

def test_attractor_basin():
	r = [
		['x1', ['x1','x2'], [1,0,0,0]],
		['x2', ['x1','x2'], [1,1,1,0]],
	]
	test_bn = dynpy.bn.BooleanNetwork(rules=r)
	atts, basins = test_bn.getAttractorsAndBasins()
	atts = [ list(tuple(i.tolist()) for i in att) for att in atts]
	basins = [ list(tuple(i.tolist()) for i in b) for b in basins]

	print(atts)
	print(basins)
	assert(atts == [[(0,1),],[(0,0),],[(1,1),],])
	assert(basins == [[(0, 1), (1, 0)], [(0, 0)], [(1, 1)]])
