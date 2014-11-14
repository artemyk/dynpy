from __future__ import division, print_function, absolute_import
import six
range = six.moves.range
map = six.moves.map

import dynpy
import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import raises

densemx = dynpy.mx.DenseMatrix.format_mx([[0,1],[2,3]])
sparsemx = dynpy.mx.SparseMatrix.format_mx([[0,1],[2,3]])

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
	
def test_issparse():
	assert( dynpy.mx.issparse(densemx) == False)
	assert( dynpy.mx.issparse(sparsemx) == True)

@raises(ValueError)
def test_issparse_invalid():
	assert( dynpy.mx.issparse('not a matrix') )

def test_todense():
	assert( dynpy.mx.issparse(dynpy.mx.todense(densemx)) == False)
	assert( dynpy.mx.issparse(dynpy.mx.todense(sparsemx)) == False)

def test_tosparse():
	assert( dynpy.mx.issparse(dynpy.mx.tosparse(densemx)) == True)
	assert( dynpy.mx.issparse(dynpy.mx.tosparse(sparsemx)) == True)

def test_hashable():
	# 1-dimensional
	hash(dynpy.mx.hashable_array(np.ravel(densemx[0,:])))
	hash(dynpy.mx.hashable_array(densemx[0:1,:]))

	# 2-dimensional
	hash(dynpy.mx.hashable_array(densemx[0,:]))

	# large 2-dimensional
	hash(dynpy.mx.hashable_array(np.zeros((1500,1500))))

def test_arrayequal():
	assert( dynpy.mx.array_equal(densemx, densemx) )
	assert( not dynpy.mx.array_equal(densemx, 1+densemx) )

def test_getdiag():
	assert( (dynpy.mx.getdiag(densemx) == [0,3]).all() )
