import numpy as np
import numpy.linalg
import scipy.linalg
import scipy.sparse as ss
import scipy.sparse.linalg
import hashlib

def toarray(mx):
    if isinstance(mx, np.ndarray):
        return mx
    else:
        return mx.toarray()

def hash_np(mx):
    return hashlib.sha1(mx).hexdigest()



class MxBase(object):

    @classmethod
    def finalizeMx(cls, mx):
        if not np.allclose(mx.sum(axis=1), 1.0):
            raise Exception('State transitions do not add up to 1.0')
        return mx

    @classmethod
    def getLargestRightEigs(cls, mx):
        raise NotImplementedError  # this is a virtual class, sublcasses should implement

    @classmethod
    def getLargestLeftEigs(cls, mx):
        vals, vecs = cls.getLargestRightEigs(mx.T)
        return vals, vecs.T

    @classmethod
    def pow(cls, mx, exponent):
        """Raise matrix to a power


        """
        raise NotImplementedError  # this is a virtual class, sublcasses should implement

    @classmethod
    def expm(cls, mx):
        """Matrix exponential"""
        raise NotImplementedError  # this is a virtual class, sublcasses should implement


class SparseMatrix(MxBase):
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
        #    raise Exception('Transition matrix for this class should not be sparse')
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

