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


import mx

# Constants for finding attractors
MAX_ATTRACTOR_LENGTH = 5
TRANSIENT_LENGTH     = 30

DEFAULT_TRANSMX_CLASS = mx.SparseMatrix

class DynamicalSystemBase(object):

    def __init__(self, discrete_time = True):
        self.discrete_time = discrete_time

    def iterateState(self, startState, num_iters=1):
        NotImplementedError

    def checkTransitionMatrix(self):
        if self.trans.shape[0] != self.trans.shape[1]:
            raise Exception('Expect square transition matrix (got %s)' % self.trans.trans.shape)

    def getTrajectory(self, startState, num_iters=1):
        cState = startState
        returnStates = [cState,]
        for i in range(num_iters):
            cState = self.iterateState(cState)
            returnStates.append(cState)
        return returnStates

    def stg_igraph(self):
        return igraph.Graph(zip(*self.trans.nonzero()), directed=True)

    def getAttractorsAndBasins(self):

        STG = self.stg_igraph()

        multistepDyn = self.transCls.pow(self.trans, TRANSIENT_LENGTH) 
        attractorStates = np.ravel( self.transCls.make2d( multistepDyn.sum(axis=0) ).nonzero()[1]  )

        basins = collections.defaultdict(list)
        for attState in attractorStates:
            cBasin = tuple( STG.subcomponent(attState, mode = 'IN') ) 
            basins[cBasin].append(attState)

        basinAtts = basins.values()
        basinStates = basins.keys()
        bySizeOrder = np.argsort(map(len, basinStates))[::-1]
        return [basinAtts[b] for b in bySizeOrder], [basinStates[b] for b in bySizeOrder]


class LinearDynamicalSystem(DynamicalSystemBase):
    def __init__(self, trans, transCls = DEFAULT_TRANSMX_CLASS, discrete_time = True):
        self.discrete_time = discrete_time
        self.trans = trans
        self.transCls = transCls

    def iterateState(self, startState, num_iters=1):
        curStartStates = self.transCls.formatMx( startState )
        r = curStartStates.dot( self.transCls.pow(self.trans, num_iters) )
        return r

    def equilibriumDistribution(self):
        vals, vecsL = self.transCls.getLeftEigs(self.trans)

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

    def getMultistepDynsys(self, num_iters):
        # TODO!!!!
        import copy
        rObj = copy.copy(self)
        rObj.trans = self.transCls.pow(self.trans, num_iters)
        return rObj

class DynamicalSystemEnsemble(LinearDynamicalSystem):
    def __init__(self, baseDynamicalSystem, discrete_time = True):
        self.baseDynamicalSystem = baseDynamicalSystem
        super(DynamicalSystemEnsemble, self).__init__(trans = baseDynamicalSystem.trans, transCls = baseDynamicalSystem.transCls, discrete_time = discrete_time)


    def getUniformDistribution(self):
        num_states = self.baseDynamicalSystem.trans.shape[0]
        return np.ones(num_states) / float(num_states)



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



class MultivariateDynamicalSystemBase(DynamicalSystemBase):

    def __init__(self, num_nodes, node_labels=None, transMatrixClass=DEFAULT_TRANSMX_CLASS, discrete_time=True):
        self.num_nodes = num_nodes
        self.node_set = tuple(range(self.num_nodes))
        if node_labels is None:
            node_labels = range(self.num_nodes)
        self.node_labels = tuple(node_labels)  # tuple(map(str, node_labels))
        self.node_label_ndxs = dict((l, ndx) for ndx, l in enumerate(self.node_labels))
        self.transCls = transMatrixClass
        super(MultivariateDynamicalSystemBase, self).__init__(discrete_time)
