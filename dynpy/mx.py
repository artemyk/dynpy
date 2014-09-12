"""Module which provides a consistent interface for working with both dense 
arrays and sparse matrices. Also constains some utility functions for working 
with matrices.
"""
from __future__ import division, print_function, absolute_import
import sys
if sys.version_info >= (3, 0):
    xrange = range

import numpy as np
import numpy.linalg
import scipy.linalg
import scipy.sparse as ss
import scipy.sparse.linalg
import hashlib


def toarray(mx):
    """Convert `mx` to np.ndarray type if it is not that already
    """
    if isinstance(mx, np.ndarray):
        return mx
    else:
        return mx.toarray()

def todense(mx):
    """Convert `mx` to a dense format, if it is not that already
    """
    if ss.issparse(mx):
        return np.asarray(mx.todense())
    else:
        return np.asarray(mx)

def hash_np(mx):
    """Provide a hash value for matrix or array (useful for using them as 
    dictionary keys, for example)
    """
    return hashlib.sha1(mx).hexdigest()



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
    def toDense(cls, mx):
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
    def formatMx(cls,mx):
        mx = ss.csc_matrix(mx)
        return mx

    @classmethod
    def finalizeMx(cls,mx):
        #if not ss.issparse(mx):
        #    raise Exception('Transition matrix for this class should be sparse')
        return cls.formatMx(mx)

    @classmethod
    def pow(cls, mx, exponent):
        rMx = ss.eye(m=mx.shape[0])
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
    def toDense(cls, mx):
        return mx

