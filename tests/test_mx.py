from __future__ import division, print_function, absolute_import
import six
range = six.moves.range
map = six.moves.map

import dynpy
import numpy as np
from numpy.testing import assert_array_equal

def test_from_coords_dense():
	rows = np.array([0,0,0])
	cols = np.array([0,0,0])
	data = np.array([1,1,1])
	r = dynpy.mx.DenseMatrix.from_coords(rows, cols, data, shape=(2,2))

	assert_array_equal(r, np.array([[3,0],[0,0]]))

def test_from_coords_sparse():
	rows = np.array([0,0,0])
	cols = np.array([0,0,0])
	data = np.array([1,1,1])
	r = dynpy.mx.SparseMatrix.from_coords(rows, cols, data, shape=(2,2))
	r = r.todense()

	assert_array_equal(r, np.array([[3,0],[0,0]]))

def test_array_equal():
	a = np.array([0,])
	b = np.array([0,])
	c = np.array([1,])
	assert( dynpy.mx.DenseMatrix.array_equal(a, b) )
	assert( not dynpy.mx.DenseMatrix.array_equal(a, c) )
	