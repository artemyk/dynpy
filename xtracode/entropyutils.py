
def getCondEntropy(dynObjTrans, startStateDist=None):
  if startStateDist is None:
    NotImplementedError
    startStateDist=dynObj.startStateProbs
  if startStateDist is None:
    startStateDist=np.array( [1./(2**dynObj.num_nodes),] * 2**dynObj.num_nodes )
  e = 0.
  if mx.issparse(dynObjTrans):
    #if num_iters != 1: NotSupportedError
    # startStates, endStates, conds = mx.getSparseMatrixRowColData(dynObj.trans)
    d2 = dynObjTrans.tocoo()
    startStates, endStates, conds = d2.row, d2.col, d2.data
    
    gNdxs = conds > 0.
    
    jDist = startStateDist[0,startStates][gNdxs] * conds
    #cDist = multTransMatrix(startStates, dynObj.trans)
      
    return -1 * np.sum( jDist * np.log2(conds) )
    
  else:
    # raise_matrix_power(dynObj.trans)
    jDist = (np.ravel(startStateDist) * dynObjTrans.T).T
    gNdxs = jDist>0
    return -1 *  np.sum( jDist[gNdxs] * np.log2(dynObjTrans[gNdxs]) ) 
    
  return e
  

def entropy(d):
  cD = d.data if mx.issparse(d) else d.flat
  cD = cD[cD!=0]
  return -1*np.sum(cD * np.log2(cD))
  
