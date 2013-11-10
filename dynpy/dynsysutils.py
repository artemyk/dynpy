
# VARIOUS UTILITY FUNCTIONS, SORT OUT

def convertNetToWeightedNet(edges):  
    return edges if type(edges) is dict else dict(zip( edges, [1.0,] * len(edges) ))

def convertNetToDynObj(num_nodes, edges, title, equilibriumStartStateProbs = False):
    # edges can be list (for unweighted) like [(0,1), (4,5)]
    # or dict for weighted: { (0,1): 0.5, (4,5): 0.10 }
    
    edgesDict = convertNetToWeightedNet(edges)

    allVerts = set([e[0] for e in edgesDict] + [e[1] for e in edgesDict])
    vertRemap = dict( (v,ndx) for ndx,v in enumerate(sorted(allVerts)) )
    #num_nodes = max( allVerts ) + 1
    outK = dict([(v,0) for v in allVerts])
    transMatrix = np.zeros( (num_nodes, num_nodes) )
    for (fromEdge, toEdge), weight in edgesDict.iteritems(): 
      outK[fromEdge] += weight
      outK[toEdge] += weight
    for (fromV, toV), weight in edgesDict.iteritems(): 
      transMatrix[vertRemap[fromV],vertRemap[toV]] = weight/outK[fromV]
      transMatrix[vertRemap[toV],vertRemap[fromV]] = weight/outK[toV]
    
    if equilibriumStartStateProbs:
      equilibriumDist = getEquilibriumDistribution(transMatrix)
    else:
      equilibriumDist = np.ones(num_nodes) * 1./num_nodes
      
    stateMap = {}
    for n in allVerts:
      cState = vertRemap[n]
      oneBit = [0,] * num_nodes ; oneBit[cState] = 1
      stateMap[tuple2int(oneBit)] = cState
    dObj = dynamicsObj(num_nodes = num_nodes, title=title, node_labels=map(str,allVerts), 
                       trans=mx.mxClass(transMatrix), startStateProbs=mx.mxClass(equilibriumDist), startStateMap=stateMap, endStateMap=stateMap) 
    return dObj


def getUniformStartStates(num_nodes):
    return mx.mxClass([1./(2**num_nodes),]*(2**num_nodes))
    
  
def dictToMatrixDyn(num_nodes, dyn):
  t=mx.getBuildMxClass()( (2**num_nodes,2**num_nodes) )
  for s, ns in dyn.iteritems():
    for n, p in ns.iteritems():
      if p == 0.: continue
      t[tuple2int(s),tuple2int(n)]=p
  return mx.mxClass(t)

def getUniformPriorParams(trans, alpha = 0.5):
  if mx.issparse(trans):
    a = trans.copy()
    a.data = 0*a.data + alpha
    return a
  else:
    return mx.mxClass(mx.getBuildMxClass()(trans.shape) + alpha)
 
import scipy.sparse as sp
def getZerosCreator(mx):
  if sp.issparse(mx):
    return sp.coo_matrix 
  else:
    return np.zeros

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

      
def getMargStateFunc(num_nodes, keep_nodes):
  flipped_keep_nodes = [(ndx, num_nodes-d-1) for ndx, d in enumerate(keep_nodes)]
  #margState = lambda state: sum( ((state >> d) & 1) << ndx for ndx, d in flipped_keep_nodes )
  margState = lambda states: sum( np.left_shift(np.bitwise_and(np.right_shift(states, d), 1), ndx) for ndx, d in flipped_keep_nodes )
  return margState      
