# dynamic systems functions

import collections
import operator
import numpy as np
import scipy.sparse as ss
import scipy.linalg
import scipy.sparse.linalg
import igraph

import sys
if sys.version_info < (3,):
    range = xrange


import mxutils

# Constants for finding attractors
MAX_ATTRACTOR_LENGTH = 5
TRANSIENT_LENGTH     = 30


class StochasticDynSysBase(object):

    def __init__(self, trans):
        self.trans      = trans

    @classmethod
    def getRightEigs(cls, mx):
        NotImplementedError

    @classmethod
    def getLeftEigs(cls, mx):
        vals, vecs = cls.getRightEigs(mx.T)
        return vals, vecs.T

    def equilibriumDistribution(self):
        if self.trans.shape[0] != self.trans.shape[1]:
            raise Exception('Expect square transition matrix (got %s)' % self.trans.shape)

        vals, vecsL = self.getLeftEigs(self.trans)

        oneEigenvals = np.flatnonzero(np.abs(vals - 1) < 1e-8)
        if len(oneEigenvals) != 1:
            raise Exception("Expected one eigenvalue of 1, but found %d instead (%s)" %
                            (len(oneEigenvals), oneEigenvals))

        equilibriumDistribution = np.real_if_close(np.ravel(vecsL[:, oneEigenvals]))
        if np.any(np.iscomplex(equilibriumDistribution)):
            raise Exception("Expect equilibrium distribution to be real! %s" % equilibriumDistribution)
        equilibriumDistribution = equilibriumDistribution / np.sum(equilibriumDistribution)
        if np.any(equilibriumDistribution < 0):
            raise Exception("Expect equilibrium distribution to be positive!")

        return equilibriumDistribution

    @classmethod
    def getTransitionList(cls, mx):
        return zip(*mx.nonzero())
    def stg_igraph(self):
        return igraph.Graph(self.getTransitionList(self.trans), directed=True)

    def getAttractorsAndBasins(self):

        STG = self.stg_igraph()

        multistepDyn = mxutils.raise_matrix_power(self.trans, TRANSIENT_LENGTH) 
        attractorStates = np.ravel( mxutils.make2d( multistepDyn.sum(axis=0) ).nonzero()[1]  )

        basins = collections.defaultdict(list)
        for attState in attractorStates:
            cBasin = tuple( STG.subcomponent(attState, mode = 'IN') ) 
            basins[cBasin].append(attState)

        basinAtts = basins.values()
        basinStates = basins.keys()
        bySizeOrder = np.argsort(map(len, basinStates))[::-1]
        return [basinAtts[b] for b in bySizeOrder], [basinStates[b] for b in bySizeOrder]

    """        

    def getAttractorDistribution(self, transientLength=TRANSIENT_LENGTH, maxAttractorLength=MAX_ATTRACTOR_LENGTH):
        # ""
        transMx = self.trans

        bMatrix = mxutils.raise_matrix_power(transMx, TRANSIENT_LENGTH)
        allAttractors = 0.
        allCol = []
        allRow = []
        allData = []

        for i in range(MAX_ATTRACTOR_LENGTH):
            bMatrix = mxutils.mxDot(bMatrix, transMx)
            print bMatrix.shape, 
            asdf
            allAttractors = allAttractors + bMatrix  # mx.toDense(bMatrix)
        #""
        cStateDistribution = self.iterateDyn(num_iters = TRANSIENT_LENGTH)
        allAttractors = 0.
        for i in range(MAX_ATTRACTOR_LENGTH):
            cStateDistribution = self.iterateDyn(startStates = cStateDistribution)
            allAttractors = allAttractors + cStateDistribution
        return allAttractors / float(MAX_ATTRACTOR_LENGTH)
    """
    def iterateDyn(self, startStates=None, num_iters=1):
        assert startStates is None or startStates.shape
        curStartStates = self.mxFormat( startStates if startStates is not None else self.getUniformStartStates() )
        r = curStartStates.dot( mxutils.raise_matrix_power(self.trans, num_iters) )
        return r

    def getMultistepDynsys(self, num_iters):
        import copy
        rObj = copy.copy(self)
        rObj.trans = mxutils.raise_matrix_power(self.trans, num_iters)
        return rObj
        
    @classmethod
    def finalizeTransMx(cls, mx):
        if np.any(mx.sum(axis=1) != 1.0):
            raise Exception('State transitions do not add up to 1.0')
        return mx

class StochasticDynSys(StochasticDynSysBase): # For dense transition matrices
    @classmethod
    def mxFormat(cls, mx):
        return np.array(mx)

    def getUniformStartStates(self):
        return np.ones( shape = (self.trans.shape[0],) ) / self.trans.shape[0]

    @classmethod
    def getRightEigs(cls, mx):
        vals, vecsR = scipy.linalg.eig(mx)
        return vals, vecsR

    @classmethod
    def getEditableBlankTransMx(cls, num_states):
        return np.zeros((num_states, num_states))

    # Convert transition matrix to finalized format
    @classmethod
    def finalizeTransMx(cls, mx):
        if ss.issparse(mx):
            raise Exception('Transition matrix for this class should not be sparse')
        return super(StochasticDynSys, cls).finalizeTransMx(cls.mxFormat(mx))


class SparseStochasticDynSys(StochasticDynSys):
    @classmethod
    def mxFormat(cls, mx):
        return ss.csr_matrix(mx)

    @classmethod
    def getRightEigs(cls, mx):
        vals, vecsR = scipy.sparse.linalg.eigs(self.trans.T)
        return vals, vecsR

    @classmethod
    def getEditableBlankTransMx(cls, num_states):
        return ss.lil_matrix((num_states, num_states))

    # Convert transition matrix to finalized format
    @classmethod
    def finalizeTransMx(cls, mx):
        if not ss.issparse(mx):
            raise Exception('Transition matrix for this class should be sparse')
        return super(StochasticDynSys, cls).finalizeTransMx(cls.mxFormat(mx))

"""
class MappedStochasticStateDynSys(StochasticDynSys):
    def getUniformStartStates(num_nodes):
        NotImplementedError

    def state2Index(self, state):
        if state not in self.stateMap:
            ix = len(self.stateMap)
            self.stateMap[state] = ix
            self.invStateMap[ix] = state
        return self.stateMap[state]

    def index2State(self, index):
        return self.invStateMap[ix]

iterate:
                for i in range(num_iters):
                if False:
                    for n1 in range(curStartStates.shape[1]):
                        c2States = np.zeros(curStartStates.shape)
                        c2States[0, n1] = 1.
                        n2 = c2States.dot(self.trans)
                        g = np.nonzero(n2)[1]
                        if len(g) != 1:
                            print "NO NEXT?!?"
                        if n1 == self.invEndStateMap[g[0]]:
                            import setfuncs
                            # print n1, "-->", self.invEndStateMap[g[0]]
                            print "SELF-ATT:", n1, setfuncs.bits(n1, self.num_nodes)

                nStates = np.zeros(curStartStates.shape)
                nStates[:, self.invEndStateMap.values()] = curStartStates.dot(self.trans)
                # print curStartStates[0,580],nStates[0,580], curStartStates.dot(self.trans)[0,98]
                curStartStates = nStates

    def __init__(self, startStateMap, endStateMap, *kargs, **kwargs):
        super(SparseStateDynamicalSystem, self).__init__(*kargs, **kwargs)

        self.startStateMap    = collections.OrderedDict(sorted(startStateMap.items(), key=operator.itemgetter(1)))
        self.invStartStateMap = collections.OrderedDict(zip(self.startStateMap.values(), self.startStateMap.keys()))
        self.endStateMap      = collections.OrderedDict(sorted(endStateMap.items(), key=operator.itemgetter(1)))
        self.invEndStateMap   = collections.OrderedDict(zip(self.endStateMap.values(), self.endStateMap.keys()))

    def getMultistepDynsys(self, num_iters):
        NotImplementedError
                """
