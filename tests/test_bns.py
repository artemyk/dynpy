import dynpy
import numpy as np

def test_def_bns():

	r = [ 
		['x1', ['x1','x2'], [1,0,0,0]],
		['x2', ['x1','x2'], [1,1,1,0]],
	]
	bn1 = dynpy.bn.BooleanNetwork(rules = r)

	r2 = [ 
		['x1', ['x1','x2'], lambda x1,x2: (x1 and x2) ],
		['x2', ['x1','x2'], lambda x1,x2: (x1 or  x2) ],
	]
	bn2 = dynpy.bn.BooleanNetwork(rules = r)

	assert((bn1.trans-bn2.trans).max() == 0.0 )
	assert((bn1.trans-bn2.trans).min() == 0.0 )


def test_ndx2state_mx():
	bn2 = dynpy.bn.BooleanNetwork(dynpy.sample_nets.yeast_cellcycle_bn)
	assert(all(bn2.ndx2stateMx[0,:] == np.zeros(bn2.num_vars)))
	assert(all(bn2.ndx2stateMx[-1,:] == np.ones(bn2.num_vars)))
