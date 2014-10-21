from __future__ import division, print_function, absolute_import
import six
range = six.moves.range

import dynpy
import numpy as np
from numpy.testing import assert_array_equal

class TestBNs:

	def setUp(self):
		r = [
			['x1', ['x1','x2'], [1,0,0,0]],
			['x2', ['x1','x2'], [1,1,1,0]],
		]
		self.testbn = dynpy.bn.BooleanNetwork(rules=r)

	def test_def_bns(self):
		bn1 = dynpy.markov.MarkovChain.from_deterministic_system(self.testbn)

		r2 = [
			['x1', ['x1','x2'], lambda x1,x2: (x1 and x2) ],
			['x2', ['x1','x2'], lambda x1,x2: (x1 or  x2) ],
		]
		bn2base = dynpy.bn.BooleanNetwork(rules=r2, mode='FUNCS')
		bn2 = dynpy.markov.MarkovChain.from_deterministic_system(bn2base)

		assert((bn1.transition_matrix-bn2.transition_matrix).max() == 0.0 )
		assert((bn1.transition_matrix-bn2.transition_matrix).min() == 0.0 )

	def test_attractor_basin(self):
		atts, basins = self.testbn.get_attractor_basins()
		atts = [ list(tuple(i.tolist()) for i in att) for att in atts]
		basins = [ list(tuple(i.tolist()) for i in b) for b in basins]

		assert(atts == [[(0,1),],[(0,0),],[(1,1),],])
		assert(basins == [[(0, 1), (1, 0)], [(0, 0)], [(1, 1)]])

	def test_structural_graph(self):
		# ring topology
		r = [
			['x1', ['x2'], lambda i: i ],
			['x2', ['x3'], lambda i: i ],
			['x3', ['x1'], lambda i: i ],
		]
		cur_bn = dynpy.bn.BooleanNetwork(rules=r, mode='FUNCS')

		expected_graph = np.array([[0,0,1],[1,0,0],[0,1,0]])

		G = cur_bn.get_structural_graph()
		assert_array_equal(G, expected_graph)
