from __future__ import division, print_function, absolute_import
import six
range = six.moves.range

import numpy as np
import scipy.sparse as ss
import dynpy

class ExampleSystem(dynpy.dynsys.DiscreteStateDynamicalSystem):

	def states(self):
		return [1,2]

	def _iterate_1step_discrete(self, x):
		return 2 if x == 1 else 2


def test_dynsys():
	d = ExampleSystem()
	d.print_attractor_basins()
