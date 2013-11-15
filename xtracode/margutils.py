
      
def getMargStateFunc(num_vars, keep_nodes):
  flipped_keep_nodes = [(ndx, num_vars-d-1) for ndx, d in enumerate(keep_nodes)]
  #margState = lambda state: sum( ((state >> d) & 1) << ndx for ndx, d in flipped_keep_nodes )
  margState = lambda states: sum( np.left_shift(np.bitwise_and(np.right_shift(states, d), 1), ndx) for ndx, d in flipped_keep_nodes )
  return margState    



    def marginalizeDynamics(self, nodes, startStateProbs = None):
        nodes = tuple(nodes)
        if len(nodes) == self.num_nodes - 1:
            key = (nodes,)
            if key not in self._marginalCache:
                newTrans = self.rawMarginalize(self.trans, nodes, startStateProbs)
                newPrior = self.rawMarginalize(self.priorObj, nodes, startStateProbs)
                self._marginalCache[key] = self.__class__(self.title, len(nodes), newTrans, newPrior)
            return self._marginalCache[key]
        else:
            # Setup hierarchy of marginalization
            for n in self.node_set:
                if n not in nodes: 
                    nNodes = sorted(nodes + (n,))
                    nNdxs = [ndx for ndx, n in enumerate(nNodes) if n in nodes]
                    return self.marginalizeDynamics(
                            nNodes, startStateProbs).marginalizeDynamics(
                                nNdxs, startStateProbs)


    def rawMarginalize(self, transTable, nodes, startStateProbs = None):
        if startStateProbs is None: startStateProbs = getUniformStartStates(transTable)
        marginalStartStateProbs = marginalizeDist(startStateProbs, nodes)
        newTrans = {}
        for startState, startStateProb in startStateProbs.iteritems():
            nextStates = transTable[startState]
            k = tuple(startState[node] for node in nodes)
            if k not in newTrans: newTrans[k] = {}
            for nextState, nextStateProb in nextStates.iteritems():
                k2 = tuple(nextState[node] for node in nodes)
                newTrans[k][k2] = newTrans[k].get(k2, 0.) + nextStateProb * (startStateProbs[startState] / marginalStartStateProbs[k])
        return newTrans
                
    def rawMarginalize(self, trans, nodes, startStateProbs = None):
        if nodes == self.node_set: return trans
        
        if startStateProbs is None: startStateProbs = getNumpyUniformStartStates(self)
        cDist = np.multiply(trans , startStateProbs)
        cDist = marginalizeNumpyDist(cDist, list(nodes) + [self.num_nodes + n for n in nodes])
        
        conditioningDist = marginalizeNumpyDist(cDist, np.array(xrange(len(nodes))))
        return cDist / conditioningDist
        
    def iterateDyn(self, startStates = None, num_iters = 1):
        curStartStates = copy.deepcopy(startStates) if startStates is not None else getUniformStartStates(self.trans)
        for i in xrange(num_iters):
            curStartStates  = curStartStates.dot(trans)
        return curStartStates




def marginalizeNumpyDist(dist,nodes):
    #print "Marginalizing over all not in ", nodes
    rD = np.copy(dist)
    dims = range(len(dist.shape))[::-1]
    for cDim in dims:
        if cDim in nodes: continue
        rD = np.sum(rD,axis=cDim)
    return rD

def marginalizeDist(dist, nodes):
    nDist = {}
    for s, prob in dist.iteritems():
        k = tuple(s[node] for node in nodes)
        nDist[k] = nDist.get(k, 0) + prob
    return nDist  