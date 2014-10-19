"""Module which provides a consistent interface for working with both dense
arrays and sparse matrices. Also constains some utility functions for working
with matrices.
"""
from __future__ import division, print_function, absolute_import
import six
range = six.moves.range

import numpy as np
import numpy.linalg
import scipy.linalg
import scipy.sparse as ss
import scipy.sparse.linalg

import hashlib
import functools

@functools.total_ordering
class hashable_array(np.ndarray):
    """This class provides a hashable and sortable np.array.  This is useful for 
    using np.array as dicitionary keys, for example.

    Notice that hashable arrays change the default behavior of numpy arrays for 
    equality and comparison operators.  Instead of performing element-wise
    tests, hashable arrays return a value for the array as a whole
    """

    def __new__(cls, data): 
        return np.array(data, copy=False).view(type=cls)

    def __init__(self, values): 
        self.__hash = int(hashlib.sha1(self).hexdigest(), 16)

    def __hash__(self):
        return self.__hash

    def __eq__(self, other): # equality test
        return np.array_equal(self, other)

    def __lt__(self, other):  # comparison test
        if self.size < other.size:
            return True
        if self == other:
            return False
        nonequal = ~np.equal(self, other)
        return np.less(self[nonequal], other[nonequal])[0]

class MxBase(object):
    """Base class from which sparse and dense matrix operation classes inherit
    """
    @classmethod
    def createEditableZerosMx(cls, shape):
        """Create blank editable transition matrix, of size specified by `shape`
        """
        raise NotImplementedError  # virtual class, sublcasses should implement

    @classmethod
    def formatMx(cls,mx):
        """Format a matrix `mx` into the current class's preferred matrix type
        (i.e., convert a dense matrix to sparse, or vice-versa, as appropriate)
        """
        raise NotImplementedError  # virtual class, sublcasses should implement

    @classmethod
    def finalizeMx(cls, mx):
        """Finalize processing of editable transition matrix `mx`
        """
        pass

    @classmethod
    def getLargestRightEigs(cls, mx):
        """Get largest right eigenvectors and eigenvalues of matrix  `mx`
        """
        raise NotImplementedError  # virtual class, sublcasses should implement

    @classmethod
    def getLargestLeftEigs(cls, mx):
        """Get largest left eigenvectors and eigenvalues of matrix  `mx`

        Returns
        -------
        1-dimensional numpy array
            Eigenvalues
        2-dimensional numpy array
            Eigenvectors
        """
        vals, vecs = cls.getLargestRightEigs(mx.T)
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


class SparseMatrix(MxBase):
    """Class for sparse matrix operations.  See documentation for
    :class:`dynpy.mx.MxBase` for description of methods.
    """
    @classmethod
    def createEditableZerosMx(cls, shape):
        return ss.lil_matrix(shape)

    @classmethod
    def formatMx(cls, mx):
        mx = ss.csc_matrix(mx)
        return mx

    @classmethod
    def finalizeMx(cls, mx):
        #if not ss.issparse(mx):
        #    raise Exception('Transition matrix for this class should be sparse')
        return cls.formatMx(mx)

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
    def getLargestRightEigs(cls, mx):
        vals, vecsR = scipy.sparse.linalg.eigs(mx, k=mx.shape[0]-2, which='LR')
        return vals, vecsR

    @classmethod
    def make2d(cls, mx):
        return mx

    @classmethod
    def toDense(cls, mx):
        return mx.todense()


class DenseMatrix(MxBase):
    """Class for dense matrix operations.  See documentation for
    :class:`dynpy.mx.MxBase` for description of methods.
    """
    @classmethod
    def expm(cls, mx):
        return scipy.linalg.expm(mx)

    @classmethod
    def getLargestRightEigs(cls, mx):
        vals, vecsR = scipy.linalg.eig(mx, right=True, left=False)
        vals, vecsR = scipy.linalg.eig(mx, right=True, left=False)
        return vals, vecsR

    @classmethod
    def createEditableZerosMx(cls, shape):
        return np.zeros(shape)

    @classmethod
    def formatMx(cls, mx):
        if ss.issparse(mx):
            mx = mx.todense()
        mx = np.array(mx)
        return mx

    # Convert transition matrix to finalized format
    @classmethod
    def finalizeMx(cls, mx):
        #if ss.issparse(mx):
        #    raise Exception('Trans mx for this class should not be sparse')
        return cls.formatMx(mx)

    @classmethod
    def pow(cls, mx, exponent):
        return numpy.linalg.matrix_power(mx, exponent)

    @classmethod
    def make2d(cls, mx):
        return np.atleast_2d(mx)

    @classmethod
    def todense(cls, mx):
        return mx


def get_cls(mx):
    if ss.issparse(mx):
        return SparseMatrix
    else:
        return DenseMatrix

def formatMx(mx):
    return get_cls(mx).formatMx(mx)

def finalizeMx(mx):
    return get_cls(mx).finalizeMx(mx)

def pow(mx, exponent):
    return get_cls(mx).pow(mx, exponent)

def expm(mx):
    return get_cls(mx).expm(mx)

def make2d(mx):
    return get_cls(mx).make2d(mx)

def todense(mx):
    return get_cls(mx).todense(mx)

def getLargestRightEigs(mx):
    return get_cls(mx).getLargestRightEigs(mx)

def getLargestLeftEigs(mx):
    return get_cls(mx).getLargestLeftEigs(mx)

