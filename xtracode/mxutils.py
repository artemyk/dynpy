import numpy as np
import scipy.sparse as ss
issparse = ss.issparse



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