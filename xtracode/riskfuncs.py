 
    
    def partitionRisk(self, partition, **kwargs):
        if len(partition) == 1:  
            return self.instRisk(**kwargs)
        else:
            return np.sum(np.vstack([self.getChild(module).instRisk(**kwargs) for module in partition]), axis=1)

    def getChild(self, moduleNames):
        if getattr(self, '_cachedChildren', None) is None:
            self._cachedChildren = {}
        key = tuple(moduleNames)
        if key not in self._cachedChildren:
            ndxs = [ndx for ndx, n in enumerate(self.node_labels) if n in moduleNames]
            self._cachedChildren[key] = self.__class__(
                self.title + '/' + ','.join(map(str,moduleNames)),
                moduleNames,
                self.rawMarginalize(self.trans, ndxs, self.startStateProbs),
                self.rawMarginalize(self.priorObj, ndxs, self.startStateProbs),
                marginalizeNumpyDist(self.startStateProbs, ndxs)
            )
        return self._cachedChildren[key]
        
    def instRisk(self, cumulative = True, doCausal = False, N_BINS=DEFAULT_BINS, N_MIN=DEFAULT_N_MIN,N_MAX=DEFAULT_N_MAX):
        if getattr(self, '_cachedInstRisk', None) is None:
            import time
            startTime = time.time()
            
            dataRng = getDataRng(N_MIN, N_MAX, N_BINS)
            instData = np.zeros(len(dataRng))
            #print "At a001 %0.5f" % (time.time()-startTime)
            #print "At a002 %0.5f" % (time.time()-startTime)    
                
            if doCausal: 
                NotImplemented
                causalMarginalDyns  = {} if doCausal else None
                causalMarginalDyns[nodes] = dynObj.marginalizeDynamics(nodes)
            
            #print "At a003 %0.5f" % (time.time()-startTime)    
            
            if False:
                tE = 0    
                for nodes in partition:
                    e = 0
                    for startState, nStates in marginalDyns[nodes].trans.iteritems():
                        e += 1./len(marginalDyns[nodes].trans) * entropy(nStates)
                    tE += e
                    # print nodes, e # , trainingMarginals[nodes].trans, 
                # print prettyPartition(partition, node_labels), "Stochastic interaction:", tE
            
                if False:
                    for n, dObj in ( ('Noncausal', marginalDyns), ('Causal', causalMarginalDyns)):
                        if not dObj: continue
                        print "***", n
                        for k, v in dObj.iteritems():
                            print k
                            transPrint(v.trans)
                    # sys.exit()
            
            instData = getNumpyCrutchfieldEnergy(self, dataRng, 1., self.startStateProbs)
            
            #print "At a004 %0.5f" % (time.time()-startTime)    
            
            #print partition, \
            #      sum(getNumpyCondEntropy(marginalDyns[nodes], startStateMarginals[nodes]) for nodes in partition), \
            #      instData[-1]
            self._cachedInstRisk = instData
            
        return self._cachedInstRisk 
        




cachedMCnts = {}
def getPriorMultinomialCounts(numStates, numSteps):
    global cachedMCnts
    key = (numStates,numSteps)
    if key not in cachedMCnts:
        if numSteps == 1:
            print "Generating first step for ", numStates
            a=np.random.multinomial(1, [1./numStates]*numStates, size=10000)
            ps=set((tuple(a1) for a1 in a))
            cachedMCnts[key] = ps
            # print "ps", len(ps), len(ps) * (nS ** N)
        else:
            nSet = set()
            prevCounts = getPriorMultinomialCounts(numStates, numSteps - 1)
            fCounts    = getPriorMultinomialCounts(numStates, 1)
            for pCnt in prevCounts:
                for fCnt in fCounts:
                    nSet.add(tuple(np.array(pCnt)+np.array(fCnt)))
            print "Generated ", numSteps, " steps for ", numStates, " states / len=", len(nSet)
            cachedMCnts[key] = nSet
    
    return cachedMCnts[key]
                    
cachedNextCnts = {}
def getNextMultinomialCounts(dist):
    global cachedNextCnts
    if dist not in cachedNextCnts:
        print "Generating samples for ", dist
        a=np.random.multinomial(1, dist, size=10000)
        ps=set((tuple(a1) for a1 in a))
        cachedNextCnts[dist] = ps
        # print "ps", len(ps), len(ps) * (nS ** N)        
    return cachedNextCnts[dist]
    
import operator
def getEnergy3(dynObj, N, NtoPredict, testStartStates = None, debug=False):
    """
    ps = getPriorMultinomialCounts(nS, N)
    for previousObservedDist in ps:
        for startStateNum, (startState, nStates) in enumerate(dynObj.trans.iteritems()):
            nextStateDist = getNextMultinomialCounts(nStates.values())
            
            a=np.random.multinomial(1, nV, size=1000)
            a=set((tuple(a1) for a1 in a))
            tProb = 0.
            for curDraw in a:
                genProb = reduce(operator.mul, (nV[n] ** curDraw[n] for n in xrange(len(nV))))
                tProb += genProb
                
            learnedDist = \
                dict( \
                    (i, dict( (j, dynObj.priorObj[startState][bits(j, nNodes)] + n/nS for j, n in enumerate(curObservedDist) ) \
                ) for i, s in dynObj.trans.iteritems() )            
            learnedTotlCounts    = dict ( (s, sum(c.values())) for s, c in learnedDist.iteritems() )
                
            for startState, nStates in notNormedLearnedDist.iteritems():
                L -= gammaln(learnedTotlCounts[startState])
                for nextState in nStates:
                    L += gammaln(notNormedLearnedDist[startState][nextState])            
        print startState, len(a), tProb
    """

def getNumpyUniformStartStates(dynObj):
    return np.zeros((2,)*dynObj.num_nodes, dtype='float') + 1./(2**dynObj.num_nodes)

from scipy.special import gammaln 
hi_resolution = 100000.
lo_resolution = 100.
resolution_cutoff = 2.
MAX_N = 10000.
# max(NarrayT)+NtoPredict+nCPrior
cachedGammaLnHi = gammaln(np.arange(0,resolution_cutoff,1/hi_resolution))
cachedGammaLnLo = gammaln(np.arange(resolution_cutoff,MAX_N,1/lo_resolution))


# from scipy.weave.blitz_tools import blitz_type_factories
# from scipy.weave import scalar_spec
from scipy.weave import inline, converters
#import scipy.weave.blitz_tools
def _fast_gammaln(passedIn):
    a_2d = passedIn.astype('double')
    assert(len(a_2d.shape) == 2)
    new_array = np.zeros(a_2d.shape,dtype='double')
    # NumPy_type = scalar_spec.NumPy_to_blitz_type_mapping[type]
    from scipy.special import gammaln
    support_code=\
    """
  #define M_lnSqrt2PI 0.91893853320467274178
  static double gamma_series[] = {
    76.18009172947146,
    -86.50532032941677,
    24.01409824083091,
    -1.231739572450155,
    0.1208650973866179e-2,
    -0.5395239384953e-5
  };
      """
    code = \
    r"""
  /* Lanczos method */
    for(int i = 0;i < Na_2d[0]; i++) {
        for(int j = 0;  j < Na_2d[1]; j++) {
            double x = (double) a_2d(i,j);
            //if(x <= 0.) { new_array(i,j)=0; continue; }
            //if(x != x) { new_array(i,j)=x; continue; } // nan
            // new_array(i,j) = 0.0;
            int i;
            double denom, x1, series;
            denom = x+1;
            x1 = x + 5.5;
            series = 1.000000000190015;
            for(i = 0; i < 6; i++) {
                series += gamma_series[i] / denom;
                denom += 1.0;
            }
            new_array(i,j) =  ( M_lnSqrt2PI + (x+0.5)*log(x1) - x1 + log(series/x) );
         }
     }
    """
    
    inline(code,['new_array','a_2d'], support_code=support_code, compiler='gcc', verbose=1,type_converters = converters.blitz)
    return new_array
    
def getNumpyCrutchfieldEnergy(dynObj, Narray, NtoPredict, startStates = None, causalMarginals = None, debug=False):

    import time
    startTime=time.time()
    if startStates is None: 
        tStates = getNumpyUniformStartStates(dynObj)
    else:
        tStates = np.copy(startStates)
    # print tStates
    #print "A Took %0.5f seconds" % (time.time() - startTime)
    
    if causalMarginals:
      NotImplementedError
      for startState, p in tStates.iteritems():
        nStates = dynObj.trans[startState]
        e += p * cross_entropy(causalMarginals.trans[startState], nStates)
        numNextStates = len(nStates)
      numParams += len(causalMarginals.trans) * (numNextStates - 1)
      
    else:
      #numStartStates = float(2**dynObj.num_nodes)
      #numGlobalStartStates = float(2**4) 
      
      # nCPrior = 0.5 # jeffrey's prior
      # only use jeffrey's prior, alpha=0.5 for full network ...
      # for modules, use marginalized priors ... ie, 
      nCPrior = 0.5 # priors[startState]
      NarrayT = np.atleast_2d(Narray).T
      flatStartingStateProbs = np.atleast_2d(tStates.flat)
      arr1 = flatStartingStateProbs * NarrayT + nCPrior*tStates.size
      arr2 = arr1 + NtoPredict*flatStartingStateProbs
      
      e = np.sum(gammaln(arr1) - gammaln(arr2), axis=1)
      
      #print "B Took %0.5f seconds" % (time.time() - startTime)
      
      cnt=0
      jointProbs = tStates * dynObj.trans
      
      #print "B2 Took %0.5f seconds" % (time.time() - startTime)
      
      for startState, p in np.ndenumerate(tStates):
        #cnt=cnt+1
        #if cnt % 10 == 0: print ".",
        nStates = jointProbs[startState+tuple([slice(None),]*dynObj.num_nodes)]
        prevFutures, nextFutures = 0., 0. 
        flatNextStateProbs = np.atleast_2d(nStates.flat)
        arr1 = flatNextStateProbs * NarrayT + nCPrior
        arr2 = arr1 + NtoPredict*flatNextStateProbs
        e += np.sum(gammaln(arr2)-gammaln(arr1), axis=1)
        #e += np.sum(_fast_gammaln(arr2)-_fast_gammaln(arr1), axis=1)
    
        """
        continue
        
        arr1lg = np.zeros(arr1.shape)        
        #print '05z-%0.7f'%(time.time()-a)
        arr1small = arr1<resolution_cutoff
        arr1big   = arr1>=resolution_cutoff
        #print '05a-%0.7f'%(time.time()-a)
        arr1lg[arr1small]  =cachedGammaLnHi[np.round(arr1[arr1small]*hi_resolution).astype('int')]
        #print '05b-%0.7f'%(time.time()-a)
        arr1lg[arr1big]    =cachedGammaLnLo[np.round(arr1[arr1big]*lo_resolution).astype('int')]
        #print '05c-%0.7f'%(time.time()-a)
        arr2lg = np.zeros(arr2.shape)        
        #print '05d-%0.7f'%(time.time()-a)
        arr2small = arr2<resolution_cutoff
        arr2big   = arr2>=resolution_cutoff
        #print '05e-%0.7f'%(time.time()-a)
        arr2lg[arr2small]  =cachedGammaLnHi[np.round(arr2[arr2small]*hi_resolution).astype('int')]
        #print '05f-%0.7f'%(time.time()-a)
        arr2lg[arr2big]=cachedGammaLnLo[np.round(arr2[arr2big]*lo_resolution).astype('int')]
        e += np.sum(arr2lg-arr1lg, axis=1)
        """
    #print "Took %0.5f seconds" % (time.time() - startTime)
    #print
    # print N, e
            
    # if dynObj.num_nodes==2 and numParams != 6: print "NUMPARAM:",dynObj.num_nodes, numParams, startStates
    #print
    #print Narray, NtoPredict, e
    return (-1/np.log(2))*e
def getCrutchfieldEnergy(dynObj, Narray, NtoPredict, startStates = None, causalMarginals = None, debug=False):
    e = 0.

    if startStates is None: 
        tStates = getUniformStartStates(dynObj.trans)
    else:
        tStates = copy.deepcopy(startStates)
    # print tStates
    
    if causalMarginals:
      NotImplementedError
      for startState, p in tStates.iteritems():
        nStates = dynObj.trans[startState]
        e += p * cross_entropy(causalMarginals.trans[startState], nStates)
        numNextStates = len(nStates)
      numParams += len(causalMarginals.trans) * (numNextStates - 1)
      
    else:
      from scipy.special import gammaln 
      numStartStates = float(2**dynObj.num_nodes)
      numGlobalStartStates = float(2**4) 
      for startState, p in tStates.iteritems():
        nStates = dynObj.trans[startState]
        prevFutures, nextFutures = 0., 0. 
        
        # nCPrior = 0.5 # jeffrey's prior
        # only use jeffrey's prior, alpha=0.5 for full network ...
        # for modules, use marginalized priors ... ie, 
        nCPrior = 0.5 # priors[startState]
        cPrior = 0.           
        
        for nState, nP in nStates.iteritems():
            #if nP == 0.: continue
            # print startState,'-->',nState,p,nP
            prevFutures += gammaln(Narray*p*nP+nCPrior)
            nextFutures += gammaln((Narray+NtoPredict)*p*nP+nCPrior)
            cPrior += nCPrior
            
        # if cPrior != len(tStates): print cPrior, len(tStates)
        e += gammaln(Narray*p+cPrior) - prevFutures + nextFutures - gammaln((Narray+NtoPredict)*p+cPrior)
    # print N, e
            
    # if dynObj.num_nodes==2 and numParams != 6: print "NUMPARAM:",dynObj.num_nodes, numParams, startStates
    # print
    # print Narray, NtoPredict, e
    return (-1/np.log(2))*e
    


def entropy(dist):
    return -1 * sum(p * log(p, 2) for p in dist.values() if p > 0)
def cross_entropy(dist, exp_dist):
    return -1 * sum(p * log(dist.values()[n], 2) for n,p in enumerate(exp_dist.values()) if p > 0)
def getCondEntropy(dynObj, startStates):
  e = 0.
  for startState, p in startStates.iteritems():
    nStates = dynObj.trans[startState]
    e += p * entropy(nStates)
  return e
def getNumpyCondEntropy(dynObj, startStates):
  e = 0.
  for startState, p in np.ndenumerate(startStates):
    nStates = dynObj.trans[startState+tuple([slice(None),]*dynObj.num_nodes)]
    e += p * np_entropy(nStates)
  return e
  
def np_entropy(d):
    return -1*np.sum([p* np.log2(p) for p in d.flat if p != 0.])
    
def getEnergy2(dynObj, N, NtoPredict, startStates = None, causalMarginals = None, debug=False):
    e = 0
    numParams = 0

    if startStates is None: 
        tStates = getUniformStartStates(dynObj.trans)
    else:
        tStates = copy.deepcopy(startStates)
    # print tStates
    
    if causalMarginals:
      for startState, p in tStates.iteritems():
        nStates = dynObj.trans[startState]
        e += p * cross_entropy(causalMarginals.trans[startState], nStates)
        numNextStates = len(nStates)
        
      numParams += len(causalMarginals.trans) * (numNextStates - 1)
      
    else:
        e = getCondEntropy(dynObj, tStates)
        numParams = sum( (dynObj.trans[startState] - 1) for startState in tStates)
      
    # if dynObj.num_nodes==2 and numParams != 6: print "NUMPARAM:",dynObj.num_nodes, numParams, startStates
    return e + numParams/(2 * (N+NtoPredict)) 
def getMinInstantRisk(instantRisks, minsOnly = False):
    mx = len(instantRisks[instantRisks.keys()[0]])
    # rearrange dict[partition][n] to be [n][partition]
    for n in xrange(mx):
      for partition in instantRisks.keys():
        print instantRisks[partition]
        print partition, n, instantRisks[partition][n]

    pivotedInstantRisks = [dict(zip(instantRisks.keys(), (instantRisks[partition][n] for partition in instantRisks.keys()) )) for n in xrange(mx)]
    # print pivotedInstantRisks
    oldMinPartition, newMinPartition = None, None
    minRisk, transitionPoints = np.zeros(mx), []
    
    useParts = dict.fromkeys(instantRisks.keys(), True if not minsOnly else False)
    
    for n in xrange(mx):
        newMinPartition, minRisk[n] = min(pivotedInstantRisks[n].iteritems(), key=operator.itemgetter(1))
        # print "HERE?", newMinPartition, oldMinPartition
        if oldMinPartition is None or minRisk[n] < pivotedInstantRisks[n][oldMinPartition]:
            if oldMinPartition is not None:  transitionPoints.append(n-1)
            oldMinPartition = newMinPartition
    
        if minsOnly:
            for curP, cRisk in pivotedInstantRisks[n].iteritems():
                if cRisk == minRisk[n]: useParts[curP] = True
    
    return minRisk, transitionPoints, useParts

def getTotalPartition(instantRisks):   
    return [k for k in instantRisks.keys() if len(k)==1][0]
    
def getTotalModularity(instantRisks, N_MIN=DEFAULT_N_MIN, N_MAX=DEFAULT_N_MAX, N_BINS = DEFAULT_BINS):
    # mx = len(instantRisks[instantRisks.keys()[0]])
    
    minRisk, transitionPoints, useParts = getMinInstantRisk(instantRisks)
    # find whole partition
    
    r_modular = np.cumsum(minRisk)
    r_whole   = np.cumsum(instantRisks[getTotalPartition(instantRisks)])
    return (r_whole[-1] - r_modular[-1]) * ((N_MAX-N_MIN) / float(N_BINS))

def getPlotLines(instantRisks, doModular = True, doCumulative = False, minsOnly = False, \
                 node_labels = None,                
                 dataRng = None, N_MIN=DEFAULT_N_MIN, N_MAX=DEFAULT_N_MAX, N_BINS =DEFAULT_BINS, \
                 colors = ['b','g','c','r','m','y'], hatches = [ '--', '-.', '--',':', '--','-']):

    if dataRng is None:
        dataRng = getDataRng(N_MIN, N_MAX, N_BINS)
    prependedDataRng = np.append([0], dataRng)
    binIntervals = prependedDataRng[1:]-prependedDataRng[:-1]
    cumBinInterval = np.cumsum(binIntervals)

    rLines, rLineProps = [], []
    minRisk, transitionPoints, useParts = getMinInstantRisk(instantRisks, minsOnly)
    allParts = sorted([p for p, u in useParts.iteritems() if u == True], key=len)
    print "Transition points:", transitionPoints
    for partitionNdx, partition in enumerate(allParts):
        cData = instantRisks[partition]
        if doCumulative: cData = np.cumsum(cData * binIntervals)
        rLines.append((dataRng, cData, (colors[partitionNdx%len(colors)] if colors else '') + hatches[partitionNdx % len(hatches)]))
        rLineProps.append({'label':'$Q_{' + prettyPartition(partition, node_labels)+'}$'})
        # rLineProps.append({'label':prettyPartition(partition, node_labels)})

    if doModular:        
        cData = minRisk
        if doCumulative: cData = np.cumsum(cData * binIntervals)
        rLines.append((dataRng, cData, 'k-'))
        rLineProps.append({'label':'$Q^{*}$', 'linewidth':2})
        
        # if not skipTransitionPoints:
        if len(transitionPoints):
            rLines.append(([cumBinInterval[t] for t in transitionPoints ],[cData[n] for n in transitionPoints],'ko'))
            rLineProps.append({})

    return rLines, rLineProps

def plotRisk(pTitle, instantRisks, yMax = None, plotModular = True, plotCumulative = True, minsOnly = False,\
             N_MIN=DEFAULT_N_MIN, N_MAX=DEFAULT_N_MAX, N_BINS =DEFAULT_BINS, modularHatch ='-', minPlotX = 0, \
             plotOnly = False):


    plt.figure() # figsize = ((8,4) if plotCumulative else (5,4) ))
    if pTitle is not None: plt.title(pTitle)
    if plotCumulative: plt.subplot(2, 1, 1)
    plt.xlabel('N') ; plt.ylabel('KL risk')

    rLines, rLineProps = getPlotLines(instantRisks, doCumulative = False, doModular = plotModular, minsOnly = True, N_MAX=N_MAX)     
    linePlots = []
    for n in xrange(len(rLines)):
        linePlots += plt.plot(*rLines[n], **rLineProps[n])

    if yMax is not None: plt.ylim(ymax=yMax)
    plt.legend(loc='upper right')    

    if plotCumulative: 
        plt.subplot(2, 1, 2)
        plt.xlabel('N') ; plt.ylabel('Cumulative risk')
    
        rLines, rLineProps = getPlotLines(instantRisks, doCumulative = True, doModular = plotModular, minsOnly = True, N_MAX=N_MAX)     
        linePlots2 = []
        for n in xrange(len(rLines)):
            linePlots2 += plt.plot(*rLines[n], **rLineProps[n])
        plt.legend(loc='upper right')    

    plt.show()

def getDataRng(N_MIN, N_MAX, N_BINS):
    return np.arange(N_MIN, N_MAX, float(N_MAX-N_MIN)/N_BINS)

def calcInstantRisk(dynObj, partition, startStateProbs = None, cumulative = True, doCausal = False, N_BINS=DEFAULT_BINS, N_MIN=DEFAULT_N_MIN,N_MAX=DEFAULT_N_MAX):
    dataRng = getDataRng(N_MIN, N_MAX, N_BINS)
    instData = np.zeros(len(dataRng))

    if startStateProbs is None: 
        startStateProbs = getUniformStartStates(dynObj.trans)
        
    marginalDyns        = {}
    startStateMarginals = {}
    causalMarginalDyns  = {} if doCausal else None

    for nodes in partition:
        marginalDyns[nodes]  = dynObj.marginalizeDynamics(nodes, startStateProbs)
        # testingMarginals[nodes]   = dynObj.marginalizeDynamics(nodes, testStartStateProbs)
        startStateMarginals[nodes] = marginalizeDist(startStateProbs, nodes)
        
        if doCausal: 
            causalMarginalDyns[nodes] = dynObj.marginalizeDynamics(nodes)
    
    if False:
        tE = 0    
        for nodes in partition:
            e = 0
            for startState, nStates in marginalDyns[nodes].trans.iteritems():
                e += 1./len(marginalDyns[nodes].trans) * entropy(nStates)
            tE += e
            # print nodes, e # , trainingMarginals[nodes].trans, 
        # print prettyPartition(partition, node_labels), "Stochastic interaction:", tE
    
        if False:
            for n, dObj in ( ('Noncausal', marginalDyns), ('Causal', causalMarginalDyns)):
                if not dObj: continue
                print "***", n
                for k, v in dObj.iteritems():
                    print k
                    transPrint(v.trans)
            # sys.exit()
    
    for moduleNodes in partition:
        # instData[ndx] += getEnergy2(marginalDyns[nodes], N, 1., startStateMarginals[nodes], causalMarginals = (causalMarginalDyns[nodes] if causalMarginalDyns else None)  )
        instData += getCrutchfieldEnergy(marginalDyns[moduleNodes], dataRng, 1., startStateMarginals[moduleNodes], causalMarginals = (causalMarginalDyns[moduleNodes] if causalMarginalDyns else None)  )
    
    print partition, \
          sum(getCondEntropy(marginalDyns[nodes], startStateMarginals[nodes]) for nodes in partition), \
          instData[-1]
        
    return instData, dataRng