import numpy as np
import scipy.sparse as ss


class MxBase(object):

    @classmethod
    def finalizeMx(cls, mx):
        if not np.allclose(mx.sum(axis=1), 1.0):
            raise Exception('State transitions do not add up to 1.0')
        return mx

    @classmethod
    def getRightEigs(cls, mx):
        NotImplementedError

    @classmethod
    def getLeftEigs(cls, mx):
        vals, vecs = cls.getRightEigs(mx.T)
        return vals, vecs.T


class SparseMatrix(MxBase):
    @classmethod
    def createEditableZerosMx(cls, shape):
        return ss.lil_matrix(shape)

    @classmethod
    def formatMx(cls,mx):
        mx = ss.csr_matrix(mx)
        return mx

    @classmethod
    def finalizeMx(cls,mx):
        #if not ss.issparse(mx):
        #    raise Exception('Transition matrix for this class should be sparse')
        return cls.formatMx(mx)

    @classmethod
    def pow(cls, mx, exponent):
        rMx = ss.eye(mx.shape[0])
        for i in range(exponent):
          rMx = rMx.dot(mx)
        return rMx

    @classmethod
    def getRightEigs(cls, mx):
        vals, vecsR = scipy.sparse.linalg.eigs(self.trans.T)
        return vals, vecsR

    @classmethod
    def make2d(cls, mx):
        return mx

    @classmethod
    def toDense(cls, mx):
        return mx.todense()


class DenseMatrix(MxBase):        
    @classmethod
    def getRightEigs(cls, mx):
        vals, vecsR = scipy.linalg.eig(mx)
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
        return mx ** exponent

    @classmethod
    def make2d(cls, mx):
        return np.atleast_2d(mx) 

    @classmethod
    def toDense(cls, mx):
        return mx

