"""Module which provides a consistent interface for working with both dense
arrays and sparse matrices. Also constains some utility functions for working
with matrices.
"""
from __future__ import division, print_function, absolute_import
import six
range = six.moves.range
map   = six.moves.map

import numpy as np
from numpy.core import asarray

import numpy.linalg
import scipy.linalg
import scipy.sparse as ss
import scipy.sparse.linalg

import hashlib
import functools

class MxBase(object):
    """Base class from which sparse and dense matrix operation classes inherit
    """
    @classmethod
    def create_editable_zeros_mx(cls, shape, dtype=None):
        """Create blank editable transition matrix, of size specified by `shape`
        """
        raise NotImplementedError  # virtual class, sublcasses should implement

    @classmethod
    def format_mx(cls,mx):
        """Format a matrix `mx` into the current class's preferred matrix type
        (i.e., convert a dense matrix to sparse, or vice-versa, as appropriate)
        """
        raise NotImplementedError  # virtual class, sublcasses should implement

    @classmethod
    def finalize_mx(cls, mx):
        """Finalize processing of editable transition matrix `mx`
        """
        pass

    @classmethod
    def get_largest_right_eigs(cls, mx):
        """Get largest right eigenvectors and eigenvalues of matrix  `mx`
        """
        raise NotImplementedError  # virtual class, sublcasses should implement

    @classmethod
    def get_largest_left_eigs(cls, mx):
        """Get largest left eigenvectors and eigenvalues of matrix  `mx`

        Returns
        -------
        1-dimensional numpy array
            Eigenvalues
        2-dimensional numpy array
            Eigenvectors
        """
        vals, vecs = cls.get_largest_right_eigs(mx.T)
        return vals, vecs.T

    @classmethod
    def pow(cls, mx, exponent):
        """Raise matrix `mx` to a power `exponent`

        """
        raise NotImplementedError  # virtual class, sublcasses should implement

    @classmethod
    def expm(cls, mx):
        """Return matrix exponential of matrix `mx` """
        raise NotImplementedError  # virtual class, sublcasses should implement


    @classmethod
    def make2d(cls, mx):
        """Transform matrix `mx` to be 2-dimensional"""
        raise NotImplementedError  # virtual class, sublcasses should implement

    @classmethod
    def todense(cls, mx):
        """Convert matrix `mx` to dense format"""
        raise NotImplementedError  # virtual class, sublcasses should implement

    @classmethod
    def tosparse(cls, mx):
        """Convert matrix `mx` to sparse format"""
        raise NotImplementedError  # virtual class, sublcasses should implement

    @classmethod
    def isnan(cls, mx):
        """Element-wise isnan operation on matrix elements"""
        raise NotImplementedError  # virtual class, sublcasses should implement

    @classmethod
    def array_equal(cls, mx, other_mx):
        """Return True if mx and other_mx are equal (including nan's, unlike
        numpy's array_equal
        """
        raise NotImplementedError  # virtual class, sublcasses should implement

    @classmethod
    def from_coords(cls, rows, cols, data, shape):
        """Initialize matrix using data and row, column coordinates. Duplicates
        should be summed"""
        raise NotImplementedError  # virtual class, sublcasses should implement

    @classmethod
    def get_coords(cls, mx):
        """Return matrix in terms of data and row, column coordinates"""
        raise NotImplementedError  # virtual class, sublcasses should implement

    @classmethod
    def diag(cls, data):
        """Returns matrix with diagonals set to data"""
        raise NotImplementedError  # virtual class, sublcasses should implement

    @classmethod
    def getdiag(cls, data):
        """Returns diagonals of matrix """
        raise NotImplementedError  # virtual class, sublcasses should implement

    @classmethod
    def vstack(cls, data):
        """Stack arrays vertically"""
        raise NotImplementedError  # virtual class, sublcasses should implement

    @classmethod
    def multiplyrows(cls, mx, multiplier):
        """TODO: Document"""
        return NotImplementedError

class SparseMatrix(MxBase):
    """Class for sparse matrix operations.  See documentation for
    :class:`dynpy.mx.MxBase` for description of methods.
    """
    @classmethod
    def create_editable_zeros_mx(cls, shape, dtype=None):
        return ss.lil_matrix(shape, dtype=None)

    @classmethod
    def format_mx(cls, mx):
        mx = ss.csc_matrix(mx)
        return mx

    @classmethod
    def finalize_mx(cls, mx):
        #if not ss.issparse(mx):
        #    raise Exception('Transition matrix for this class should be sparse')
        return cls.format_mx(mx)

    @classmethod
    def pow(cls, mx, exponent):
        rMx = ss.eye(mx.shape[0], mx.shape[0])
        for i in range(exponent):
          rMx = rMx.dot(mx)
        return rMx

    @classmethod
    def expm(cls, mx):
        return scipy.sparse.linalg.expm(mx)

    @classmethod
    def get_largest_right_eigs(cls, mx):
        vals, vecsR = scipy.sparse.linalg.eigs(mx, k=mx.shape[0]-2, which='LR')
        return vals, vecsR

    @classmethod
    def make2d(cls, mx):
        return mx

    @classmethod
    def todense(cls, mx):
        return mx.todense()

    @classmethod
    def tosparse(cls, mx):
        return mx

    @classmethod
    def isnan(cls, mx):
        r = mx.copy()
        r.data[:] = np.isnan(r.data)
        return r

    @classmethod
    def from_coords(cls, rows, cols, data, shape):
        return ss.coo_matrix((data, (rows, cols)), shape=shape)

    @classmethod
    def get_coords(cls, mx):
        mx = mx.tocoo()
        return mx.row, mx.col, mx.data

    @classmethod
    def diag(cls, data):
        return ss.diags(data, 0)

    @classmethod
    def getdiag(cls, mx):
        ix = np.arange(mx.shape[0], dtype='int')
        return np.ravel(todense(mx[ix, ix]))

    @classmethod
    def vstack(cls, data):
        return ss.vstack(data)

    @classmethod
    def multiplyrows(cls, mx, multiplier):
        r = cls.diag(np.ravel(multiplier)).dot(mx)
        r = cls.finalize_mx(r)
        r[np.ravel(np.isnan(multiplier)),:] = np.nan
        return r


class DenseMatrix(MxBase):
    """Class for dense matrix operations.  See documentation for
    :class:`dynpy.mx.MxBase` for description of methods.
    """
    @classmethod
    def expm(cls, mx):
        return scipy.linalg.expm(mx)

    @classmethod
    def get_largest_right_eigs(cls, mx):
        vals, vecsR = scipy.linalg.eig(mx, right=True, left=False)
        vals, vecsR = scipy.linalg.eig(mx, right=True, left=False)
        return vals, vecsR

    @classmethod
    def create_editable_zeros_mx(cls, shape, dtype=None):
        return np.zeros(shape, dtype=dtype)

    @classmethod
    def format_mx(cls, mx):
        if ss.issparse(mx):
            mx = mx.todense()
        mx = np.array(mx)
        return mx

    # Convert transition matrix to finalized format
    @classmethod
    def finalize_mx(cls, mx):
        #if ss.issparse(mx):
        #    raise Exception('Trans mx for this class should not be sparse')
        return cls.format_mx(mx)

    @classmethod
    def pow(cls, mx, exponent):
        return numpy.linalg.matrix_power(mx, exponent)

    @classmethod
    def make2d(cls, mx):
        return np.atleast_2d(mx)

    @classmethod
    def todense(cls, mx):
        return mx

    @classmethod
    def tosparse(cls, mx):
        return SparseMatrix.format_mx(mx)

    @classmethod
    def isnan(cls, mx):
        return np.isnan(mx)
        
    @classmethod
    def array_equal(cls, mx, other_mx):
        try:
            mx, other_mx = asarray(mx), asarray(other_mx)
        except:
            return False
        if mx.shape != other_mx.shape:
            return False

        both_equal = np.equal(mx, other_mx)
        both_nan   = np.logical_and(np.isnan(mx), np.isnan(other_mx))
        return bool(np.logical_or(both_equal, both_nan).all())

    @classmethod
    def from_coords(cls, rows, cols, data, shape):
        mx = cls.create_editable_zeros_mx(shape, data.dtype)
        for r, c, d in zip(rows, cols, data):
            mx[r, c] += d
        # mx[rows, cols] = data  # Doesn't sum duplicates
        return mx

    @classmethod
    def get_coords(cls, mx):
        mx = ss.coo_matrix(mx)
        return mx.row, mx.col, mx.data

    @classmethod
    def diag(cls, data):
        return np.diag(data) 

    @classmethod
    def getdiag(cls, mx):
        return np.diagonal(mx)

    @classmethod
    def vstack(cls, data):
        return np.vstack(data)

    @classmethod
    def multiplyrows(cls, mx, multiplier):
        return np.multiply(mx, multiplier)


def issparse(mx):
    if ss.issparse(mx):
        return True
    elif isinstance(mx, (np.ndarray, np.generic) ):
        return False
    else:
        raise ValueError('mx does not appear to be a matrix')

def get_cls(mx):
    if issparse(mx):
        return SparseMatrix
    else:
        return DenseMatrix

def format_mx(mx):
    return get_cls(mx).format_mx(mx)

def finalize_mx(mx):
    return get_cls(mx).finalize_mx(mx)

def pow(mx, exponent):
    return get_cls(mx).pow(mx, exponent)

def expm(mx):
    return get_cls(mx).expm(mx)

def make2d(mx):
    return get_cls(mx).make2d(mx)

def todense(mx):
    return get_cls(mx).todense(mx)

def tosparse(mx):
    return get_cls(mx).tosparse(mx)

def get_largest_right_eigs(mx):
    return get_cls(mx).get_largest_right_eigs(mx)

def get_largest_left_eigs(mx):
    return get_cls(mx).get_largest_left_eigs(mx)

def isnan(mx):
    return get_cls(mx).isnan(mx)

def array_equal(mx, other_mx):
    return get_cls(mx).array_equal(mx, other_mx)

def getdiag(mx):
    return get_cls(mx).getdiag(mx)

def multiplyrows(mx, multiplier):
    return get_cls(mx).multiplyrows(mx, multiplier)


@functools.total_ordering
class hashable_array(np.ndarray):
    """This class provides a hashable and sortable np.array.  This is useful for 
    using np.array as dicitionary keys, for example.

    Notice that hashable arrays change the default behavior of numpy arrays for 
    equality and comparison operators.  Instead of performing element-wise
    tests, hashable arrays return a value for the array as a whole
    """

    # We inherit from np.ndarray but don't want to include that documentation
    # These attributes are parsed in Sphinx's conf.py
    SPHINXDOC_INHERITED_MEMBERS = False
    SPHINXDOC_UNDOC_MEMBERS     = False

    def __new__(cls, data): 
        r = np.array(data, copy=False).view(type=cls)
        r.flags.writeable = False
        return r

    def __init__(self, values): 
        if len(values) < 1000:
            if len(values.shape) == 1:
                tpl = tuple(values.tolist())
            else:
                tpl = tuple(map(tuple, values.tolist()))
            self.__hash = hash(tpl)
        else:
            self.__hash = int(hashlib.sha1(self).hexdigest(), 16)

    def __hash__(self):
        return self.__hash

    def __eq__(self, other): # equality test
        return DenseMatrix.array_equal(self, other)

    def __lt__(self, other):  # comparison test
        if self.size < other.size:
            return True
        nonequal = ~np.equal(self, other)
        if not len(nonequal):
            return False
        firstnonequal = np.flatnonzero(nonequal)[0]
        return self[firstnonequal] < other[firstnonequal]
