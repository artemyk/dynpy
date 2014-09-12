"""Module implementing some base classes useful for implementing dynamical systems"""

from __future__ import division, print_function, absolute_import

import collections
import operator
import sys
if sys.version_info < (3,):
    range = xrange

import numpy as np

from . import mx
from . import caching

# Constants for finding attractors
MAX_ATTRACTOR_LENGTH = 5
TRANSIENT_LENGTH = 30

DEFAULT_TRANSMX_CLASS = mx.DenseMatrix
DEFAULT_STATE_DTYPE = 'float64'


class DynamicalSystemBase(object):

    """
    Base class for dynamical systems.

    Parameters
    ----------
    num_vars : int, optional
        How many variables (i.e., how many 'dimensions' or 'nodes' are in the dynamical system). Default is 1
    var_names : list, optional
        Names for the variables (optional).  Default is simply the numeric indexes of the variables.
    discrete_time : bool, optional
        Whether updating should be done using discrete (default) or continuous time dynamics.
    state_dtypes : str, optional
        A name of a NumPy datatype which should be used to store the values of each variable (default set by `DEFAULT_STATE_DTYPE`)

    """

    num_vars = None  #: The number of variables in the dynamical system

    state_dtypes = DEFAULT_STATE_DTYPE  #: The numpy data types of the states of the variables in the dynamical system

    discrete_time = True #: Whether the dynamical system obeys discrete- or continuous-time dynamics

    def __init__(self, num_vars=1, var_names=None, discrete_time=True, state_dtypes=None):

        self.num_vars = num_vars

        self.var_names = tuple(var_names if var_names is not None else range(self.num_vars))
        """The names of the variables in the dynamical system"""

        if state_dtypes is not None:
            self.state_dtypes = state_dtypes

        if discrete_time is not None:
            self.discrete_time = discrete_time

        if self.discrete_time:
            self.iterate = self._iterateDiscrete
            self.iterateOneStep = self._iterateOneStepDiscrete
        else:
            self.iterate = self._iterateContinuous

    # Make this a cached property so its not necessarily run every time a dynamical systems object is created,
    # whether we need it or not
    @caching.cached_data_prop
    def var_name_ndxs(self):
        """
        A mapping from variables names to their indexes
        """
        return dict((l, ndx) for ndx, l in enumerate(self.var_names))  # Myrecarray.dtype.names

    def iterate(self, startState, max_time):
        """
        This method runs the dynamical system for `max_time` starting from `startState`
        and returns the result.  In fact, this method is set at run-time by the constructor to either 
        `_iterateDiscrete` or `_iterateContinuous` depending on whether the dynamical system
        object is initialized with `discrete_time=True` or `discrete_time=False`.
        Thus, sub-classes should override `_iterateDiscrete` and `_iterateContinuous` instead
        of this method.  See also :meth:`dynpy.dynsys.DynamicalSystemBase.iterateOneStep`

        Parameters
        ----------
        startState : numpy array or scipy.sparse matrix
            Which state to start from
        max_time : float
            Until which point to run the dynamical system (number of iterations for discrete-time
            systems or time limit for continuous-time systems)

        Returns
        -------
        numpy array or scipy.sparse matrix
            End state
        """
        raise NotImplementedError  # this method should be re-pointed in the construtor

    def iterateOneStep(self, startState):
        """
        This method runs a discrete-time dynamical system for 1 timestep. At run-time, the construct
        either repoints this method to `_iterateOneStep` for discrete-time systems, or removes it for
        continuous time systems.

        Parameters
        ----------
        startState : numpy array or scipy.sparse matrix
            Which state to start from


        Returns
        -------
        numpy array or scipy.sparse matrix
            Iterated state
        """
        pass

    def _iterateOneStepDiscrete(self, startState, num_iters=1):
        raise NotImplementedError

    def _iterateDiscrete(self, startState, max_time=1.0):
        # TODO: Use this to generate fast weave code, or at least make faster
        if max_time == 1.0:
            return self.iterateOneStep(startState)
        elif max_time == 0.0:
            return startState
        else:
            cState = startState
            for i in range(int(max_time)):
                cState = self.iterateOneStep(cState)
            return cState

    def getTrajectory(self, startState, max_time, num_points=None, logscale=False):
        """
        This method get a trajectory (i.e. matrix ) runs a discrete-time dynamical system for 1 timestep. 

        At run-time, the construct either repoints this method to `_iterateOneStep` for discrete-time systems, 
        or removes it for continuous time systems.

        Parameters
        ----------
        startState : numpy array or scipy.sparse matrix
            Which state to start from
        max_time : float
            Until which point to run the dynamical system (number of iterations for discrete-time
            systems or time limit for continuous-time systems)
        num_timepoints : int, optional
            How many timepoints to sample the trajectory at.  In other words, how big each 'step size' is.
            By default, equal to ``int(max_time)``
        logscale : bool, optional
            Whether to sample the timepoints on a logscale or not (default)

        Returns
        -------


        """
        # TODO: would this accumulate error for continuous case?
        if num_points is None:
            num_points = int(max_time)

        if logscale:
            timepoints = np.logspace(0, np.log10(max_time), num=num_points, endpoint=True, base=10.0)
        else:
            timepoints = np.linspace(0, max_time, num=num_points, endpoint=True)

        returnTrajectory = np.zeros((len(timepoints), self.num_vars), self.state_dtypes)

        cState = startState
        returnTrajectory[0, :] = cState

        for cNdx in range(1, len(timepoints)):
            nextState = self.iterate(cState, max_time=timepoints[cNdx]-timepoints[cNdx-1])
            cState    = nextState
            returnTrajectory[cNdx, :] = mx.toarray(cState)

        return returnTrajectory


class LinearSystem(DynamicalSystemBase):
    # TODOTESTS
    """
    This class implements linear dynamical systems, whether continuous or discrete-time.  It is also
    used by :class:`dynpy.dynsys.MarkovChain` to implement Markov Chain (discrete-case) or 
    master equation (continuous-case) dynamics.

    For attribute definitions, see documentation of :class:`dynpy.dynsys.DynamicalSystemBase`.

    Parameters
    ----------
    updateOperator : numpy array or scipy.sparse matrix
        Matrix defining the evolution of the dynamical system, i.e. the :math:`\\mathbf{A}` 
        in :math:`\\mathbf{x_{t+1}} = \\mathbf{x_{t}}\\mathbf{A}` (in the discrete-time case) or
        :math:`\\dot{\\mathbf{x}} = \\mathbf{x}\\mathbf{A}` (in the continuous-time case) 
    updateCls : {:class:`dynpy.mx.DenseMatrix`, :class:`dynpy.mx.SparseMatrix`}, optional 
        Whether to use sparse or dense matrices for the update operator matrix.  Default set by `dynpy.dynsys.DEFAULT_TRANSMX_CLASS`
    var_names : list, optional
        Names for the variables (optional).  Default is simply the numeric indexes of the variables.
    discrete_time : bool, optional
        Whether updating should be done using discrete (default) or continuous time dynamics.
    state_dtypes : str, optional
        A name of a NumPy datatype which should be used to store the values of each variable (default set by `DEFAULT_STATE_DTYPE`)
    """

    def __init__(self, updateOperator, updateCls=None, var_names=None, discrete_time=True, state_dtypes=None):
        super(LinearSystem, self).__init__(num_vars=updateOperator.shape[
                                           0], var_names=var_names, discrete_time=discrete_time, state_dtypes=state_dtypes)
        self.updateOperator = updateOperator
        self.updateCls = updateCls
        if discrete_time:
            self.stableEigenvalue = 1.0
        else:
            self.stableEigenvalue = 0.0

    def equilibriumState(self):
        """
        Get equilibrium state for this dynamical system, uses eigen-decomposition

        Returns
        -------
        numpy array or scipy.sparse matrix
            Equilibrium state
        """

        vals, vecs = self.updateCls.getLargestLeftEigs(self.updateOperator)
        equilEigenvals = np.flatnonzero(np.abs(vals - self.stableEigenvalue) < 1e-8)
        if len(equilEigenvals) != 1:
            raise Exception("Expected one stable eigenvalue, but found %d instead (%s)" %
                            (len(equilEigenvals), equilEigenvals))

        equilibriumState = np.real_if_close(np.ravel(vecs[equilEigenvals, :]))
        if np.any(np.iscomplex(equilEigenvals)):
            raise Exception("Expect equilibrium state to be real! %s" % equilEigenvals)

        return self.updateCls.formatMx(equilibriumState)

    def _iterateOneStepDiscrete(self, startState):
        # For discrete time systems, one step
        r = self.updateCls.formatMx(startState).dot(self.updateOperator)
        return r

    def _iterateDiscrete(self, startState, max_time=1.0):
        # For discrete time systems
        r = self.updateCls.formatMx(startState).dot(self.updateCls.pow(self.updateOperator, int(max_time)))
        return r

    def _iterateContinuous(self, startState, max_time=1.0):
        curStartStates = self.updateCls.formatMx(startState)
        r = curStartStates.dot(self.updateCls.expm(max_time * (self.updateOperator)))
        return r

    # TODO
    # def getMultistepDynsys(self, num_iters):
    #     import copy
    #     rObj = copy.copy(self)
    #     rObj.trans = self.updateOperatorCls.pow(self.updateOperator, num_iters)
    #     return rObj


class MarkovChain(LinearSystem):

    """
    This class implements a Markov Chain over the state-transition graph of an underlying 
    dynamical system, specified by the `baseDynamicalSystem` parameter.  It maintains properties
    of the underlying system, such as the sparsity of the state transition matrix, and whether
    the system is discrete or continuous-time.  The underlying system must be an instance of
    :class:`dynpy.dynsys.DiscreteStateSystemBase` and provide a transition matrix in the form
    of a `trans` property.

    It generates a Markov chain (or, in the continuous-time case, master 
    equation) over the states of the underlying system. Each state of the underlying system
    is now assigned to a separate variable in the Markov chain system; the value of each variable
    is the probability mass on the corresponding state of the underlying system.

    For example, we can use this to derivate the dynamics of a probability of an ensemble
    of random walkers on the karate club network (this basically results in a heat-equation type system):

    >>> import dynpy
    >>> import numpy as np
    >>> rw = dynpy.graphdynamics.RandomWalker(graph=dynpy.sample_nets.karateclub_net, transCls=dynpy.mx.DenseMatrix )
    >>> rwEnsemble = dynpy.dynsys.MarkovChain(rw)
    >>> 
    >>> initState = np.zeros(rw.num_vars)
    >>> initState[ 5 ] = 1
    >>> 
    >>> trajectory = rwEnsemble.getTrajectory(initState, max_time=2)[1,1]

    It can also be done for a Boolean network:
    
    >>> import dynpy
    >>> bn = dynpy.bn.BooleanNetwork(rules=dynpy.sample_nets.yeast_cellcycle_bn)
    >>> bnEnsemble = dynpy.dynsys.MarkovChain(bn)
    >>> trajectory = bnEnsemble.getTrajectory(bnEnsemble.getUniformDistribution(), max_time=80)

    If we wish to project the state of the Markov chain back onto the activations of the
    variables in the underlying system, we can use the `ndx2stateMx` matrix of the underlying system. For example:

    >>> from __future__ import print_function
    >>> import dynpy
    >>> bn = dynpy.bn.BooleanNetwork(rules=dynpy.sample_nets.yeast_cellcycle_bn)
    >>> bnEnsemble = dynpy.dynsys.MarkovChain(bn)
    >>> final_state = bnEnsemble.iterate(bnEnsemble.getUniformDistribution(), max_time=80)
    >>> print(final_state.dot(bn.ndx2stateMx))
    [ 0.          0.05664062  0.07373047  0.07373047  0.91503906  0.          0.
      0.          0.92236328  0.          0.        ]


    Parameters
    ----------
    baseDynamicalSystem : :class:`dynpy.dynsys.DiscreteStateSystemBase`
        an object containing the underlying dynamical system over which an ensemble will be created.  

    """

    def __init__(self, baseDynamicalSystem):
        if not isinstance(baseDynamicalSystem, DiscreteStateSystemBase):
            raise Exception('Base dynamical system must be discrete state and have a trans transition matrix property')
        self.baseDynamicalSystem = baseDynamicalSystem
        super(MarkovChain, self).__init__(
            updateOperator=baseDynamicalSystem.trans, updateCls=baseDynamicalSystem.transCls, discrete_time=baseDynamicalSystem.discrete_time)

    def equilibriumState(self):
        """
        Get equilibrium state (i.e. the stable, equilibrium distribution) for this dynamical system.  Uses eigen-decomposition.

        Returns
        -------
        numpy array or scipy.sparse matrix
            Equilibrium distribution
        """

        equilibriumDist = super(MarkovChain, self).equilibriumState()
        equilibriumDist = equilibriumDist / equilibriumDist.sum()

        if np.any(self.updateCls.toDense(equilibriumDist) < 0.0):
            raise Exception("Expect equilibrium state to be positive!")
        return equilibriumDist

    def getUniformDistribution(self):
        """Gets uniform starting distribution over all system states"""
        return np.ones(self.num_vars) / float(self.num_vars)


class DiscreteStateSystemBase(DynamicalSystemBase):

    """This is a base class for discrete-state dynamical systems.  Simply put, it means there exists
    a transition matrix indicating transitions between different system states.

    Parameters
    ----------
    num_vars : int, optional
        How many variables (i.e., how many 'dimensions' or 'nodes' are in the dynamical system). Default is 1
    var_names : list, optional
        Names for the variables (optional).  Default is simply the numeric indexes of the variables.
    transCls : {:class:`dynpy.mx.DenseMatrix`, :class:`dynpy.mx.SparseMatrix`}, optional 
        Whether to use sparse or dense matrices for the transition matrix.  Default set by `dynpy.dynsys.DEFAULT_TRANSMX_CLASS`
    discrete_time : bool, optional
        Whether updating should be done using discrete (default) or continuous time dynamics.
    state_dtypes : str, optional
        A name of a NumPy datatype which should be used to store the values of each variable (default set by `DEFAULT_STATE_DTYPE`)

    """

    #: The transition matrix, either as a ``numpy.array`` (for dense representations) or
    #: ``scipy.sparse`` matrix (for sparse representations). Any subclass needs to implement.
    trans = None   #: Transition matrix of dynamical system.

    #: One of {:class:`dynpy.mx.DenseMatrix`, :class:`dynpy.mx.SparseMatrix`}, indicates
    #: whether to use sparse or dense matrices for the transition matrix.  Default set by `dynpy.dynsys.DEFAULT_TRANSMX_CLASS`
    transCls = DEFAULT_TRANSMX_CLASS

    #: ``(num_states, num_vars)``-shaped matrix which maps from integer state indexes to their representations
    #: in terms of the values of the system variables.  Any subclass needs to implement.
    ndx2stateMx = None

    def __init__(self, num_vars, var_names=None, transCls=None, discrete_time=True, state_dtypes=None):
        super(DiscreteStateSystemBase, self).__init__(num_vars=num_vars,
                                                      var_names=var_names, discrete_time=discrete_time, state_dtypes=state_dtypes)
        if transCls is not None:
            self.transCls = transCls
        self._state2ndxDict = None

    def state2ndx(self, state):
        """Function which maps from multidimensional states of variables (`state`) to single-integer state indexes.
        """
        # TODOTEST
        if self._state2ndxDict is None:
            self._state2ndxDict = dict((mx.hash_np(row), ndx) for ndx, row in enumerate(self.ndx2stateMx))

        h = mx.hash_np(state.astype(self.cDataType))
        try:
            return self._state2ndxDict[h]
        except KeyError:
            raise KeyError('%r (hash=%s)' % (state, h))

    def checkTransitionMatrix(self, trans):
        """
        Internally used function that checks the integrity/format of transition matrices.
        """
        if trans.shape[0] != trans.shape[1]:
            raise Exception('Expect square transition matrix (got %s)' % trans.shape)
        sums = mx.todense(trans.sum(axis=1))
        if self.discrete_time:
            if not np.allclose(sums, 1.0):
                raise Exception('For discrete system, state transitions entries should add up to 1.0 (%s)' % sums)
        else:
            if not np.allclose(sums, 0.0):
                raise Exception('For continuous system, state transitions entries should add up to 0.0 (%s)' % sums)

    def stgIgraph(self):
        """
        Returns
        -------
        igraph `Graph` object
            The state transition graph, in the form of an igraph Graph object
        """
        import igraph
        return igraph.Graph(zip(*self.trans.nonzero()), directed=True)

    def getAttractorsAndBasins(self):
        """
        Computes the attractors and basins of the current discrete-state dynamical system. 

        Returns
        ------- 
        basinAtts : list of lists
            A list of the the attractor states for each basin (basin order is from largest basin to smallest).

        basinStates : list of lists
            A list of all the states in each basin (basin order is from largest basin to smallest).

        """

        STG = self.stgIgraph()

        multistepDyn = self.transCls.pow(self.trans, TRANSIENT_LENGTH)
        attractorStates = np.ravel(self.transCls.make2d(multistepDyn.sum(axis=0)).nonzero()[1])

        basins = collections.defaultdict(list)
        for attState in attractorStates:
            cBasin = tuple(STG.subcomponent(attState, mode='IN'))
            basins[cBasin].append(attState)

        basinAtts = basins.values()
        basinStates = basins.keys()
        bySizeOrder = np.argsort(map(len, basinStates))[::-1]
        return [basinAtts[b] for b in bySizeOrder], [basinStates[b] for b in bySizeOrder]


if __name__ == '__main__':
    import sys
    import os
    import doctest
    sys.path = [os.path.abspath("..")] + sys.path
    verbose = True
    r = doctest.testmod(None, None, None, verbose, None)  # , doctest.NORMALIZE_WHITESPACE)
    sys.exit(r[0])
