import numpy as np
import scipy.sparse as ss
issparse = ss.issparse

def make2d(mx):
  if issparse(mx): return mx
  return np.atleast_2d(mx)
  
def raise_matrix_power(d, exponent):
  if exponent == 1: return d.copy()
  if issparse(d):
    rMx = ss.eye(d.shape[0])
    for i in range(exponent):
      rMx = rMx.dot(d)

    return rMx
    # return ss.csr_matrix( rMx )
    #if exponent == 0: 
    #  return ss.eye(d.shape[0], d.shape[1]).tocsr( )
    #return reduce(mxDot, [d,]*exponent)
    #return reduce(ss.dot, [d,]*exponent)
  else:
    if exponent == 0: return np.eye(d.shape[0], d.shape[1])
    r = np.linalg.matrix_power(d, exponent)
    return r

#def isSparse(m):
#  return issparse(m)
"""
def toDense(m):
  return np.array(m.toarray()) if issparse(m) else m
def toSparse(m):
  return ss.csr_matrix(m) # if not issparse(m) else m.tocsr()


def mxDot(a, b):
  if ( ( issparse(a) and issparse(b) ) or (not issparse(a) and not issparse(b)) ):
    return a.dot(b)
  else:
    # return mx.toDense(a).dot(mx.toDense(b))
    return toSparse(a).dot(toSparse(b))
"""