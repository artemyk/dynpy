# dynamic systems functions

import collections
import operator
import sys
if sys.version_info < (3,):
    range = xrange

import numpy as np

import mx
import caching

# Constants for finding attractors
MAX_ATTRACTOR_LENGTH = 5
TRANSIENT_LENGTH = 30

DEFAULT_TRANSMX_CLASS = mx.SparseMatrix
DEFAULT_STATE_DTYPE = 'float64'


class DynamicalSystemBase(object):

    """Base class for dynamical systems."""

    def __init__(self, num_vars=1, var_names=None, discrete_time=True, state_dtypes=None):
        """
        Create a slider from *valmin* to *valmax* in axes *ax*

        *valinit*
            The slider initial position

        *label*
            The slider label

        *valfmt*
            Used to format the slider value

        *closedmin* and *closedmax*
            Indicate whether the slider interval is closed

        *slidermin* and *slidermax*
            Used to contrain the value of this slider to the values
            of other sliders.

        additional kwargs are passed on to ``self.poly`` which is the
        :class:`matplotlib.patches.Rectangle` which draws the slider
        knob.  See the :class:`matplotlib.patches.Rectangle` documentation
        valid property names (e.g., *facecolor*, *edgecolor*, *alpha*, ...)
        """

        self.num_vars = num_vars
        self.var_names = tuple(var_names if var_names is not None else range(self.num_vars))
        self.state_dtypes = state_dtypes if state_dtypes is not None else DEFAULT_STATE_DTYPE
        self.discrete_time = discrete_time

        if self.discrete_time:
            self.iterate = self._iterateDiscrete
            self.iterateOneStep = self._iterateOneStepDiscrete
        else:
            self.iterate = self._iterateContinuous

    @caching.cached_data_prop
    def var_name_ndxs(self):
        # Make this a cached property so its not necessarily run every time a dynamical systems object is created,
        # whether we need it or not
        return dict((l, ndx) for ndx, l in enumerate(self.var_names))  # Myrecarray.dtype.names

    def iterate(self, startState, max_time):
        """
        This method runs the dynamical system for ``max_time`` starting from ``startState`` 
        and returns the result.  In fact, this method is set at run-time by the constructor to either 
        ``_iterateDiscrete`` or ``_iterateContinuous`` depending on whether the dynamical system
        object is initialized with ``discrete_time=True`` or ``discrete_time=False``.
        Thus, sub-classes should override ``_iterateDiscrete`` and ``_iterateContinuous`` instead
        of this method.  See also :meth:`dynpy.dynsys.DynamicalSystemBase.iterateOneStep`

        :param: startState, which state to start from
        :param: max_time, until which point to run the dynamical system

        """
        raise NotImplementedError  # this method should be re-pointed in the construtor

    def iterateOneStep(self, startState, max_time):
        """
        """
        pass


    def _iterateOneStepDiscrete(self, startState, num_iters=1):
        """Iterate dynamical system one or more timesLoad one or more packages into parent package top-level namespace.

       This function is intended to shorten the need to import many
       subpackages, say of scipy, constantly with statements such as

         import scipy.linalg, scipy.fftpack, scipy.etc...

       Instead, you can say:

         import scipy
         scipy.pkgload('linalg','fftpack',...)

       or

         scipy.pkgload()

       to load all of them in one call.

       If a name which doesn't exist in scipy's namespace is
       given, a warning is shown.

       Parameters
       ----------
        *packages : arg-tuple
             the names (one or more strings) of all the modules one
             wishes to load into the top-level namespace.
        verbose= : integer
             verbosity level [default: -1].
             verbose=-1 will suspend also warnings.
        force= : bool
             when True, force reloading loaded packages [default: False].
        postpone= : bool
             when True, don't load packages [default: False]

     """
        raise NotImplementedError

    def _iterateDiscrete(self, startState, max_time=1.0):
        """
        TODO: Use this to generate fast weave code, or at least make faster
        """
        if max_time == 1.0:
            return self.iterateOneStep(startState)
        elif max_time == 0.0:
            return startState
        else:
            cState = startState
            for i in range(int(max_time)):
                cState = self.iterateOneStep(cState)
            return cState

    def getTrajectory(self, startState, last_timepoint, num_points=None, logscale=False):
        # TODO: would this accumulate error for continuous case?
        if num_points is None:
            num_points = last_timepoint

        if logscale:
            timepoints = np.logspace(0, np.log10(last_timepoint), num=num_points, endpoint=True, base=10.0)
        else:
            timepoints = np.linspace(0, last_timepoint, num=num_points, endpoint=True)

        returnTrajectory = np.zeros((len(timepoints), self.num_vars), self.state_dtypes)
        cState = startState
        cTimestep = 0
        for cNdx, nextTimestep in enumerate(timepoints):
            returnTrajectory[cNdx, :] = mx.toarray(cState)
            cState = self.iterate(cState, max_time=nextTimestep - cTimestep)
            cTimestep = nextTimestep

        return returnTrajectory


class LinearSystem(DynamicalSystemBase):

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

    """
    def getMultistepDynsys(self, num_iters):
        # TODO!!!!
        import copy
        rObj = copy.copy(self)
        rObj.trans = self.updateOperatorCls.pow(self.updateOperator, num_iters)
        return rObj
    """


class DynamicalSystemEnsemble(LinearSystem):

    """
    Stochastic ensemble 

    >>> import dynpy
    >>> bn = dynpy.bn.BooleanNetwork(rules=dynpy.sample_nets.yeast_cellcycle_bn)
    >>> bnEnsemble = dynpy.dynsys.DynamicalSystemEnsemble(bn)
    >>> _  = bnEnsemble.getTrajectory(bnEnsemble.getUniformDistribution(), last_timepoint=80).dot(bn.state2ndxMx)

    """

    def __init__(self, baseDynamicalSystem):
        if not isinstance(baseDynamicalSystem, DiscreteStateSystemBase):
            raise Exception('Base dynamical system must be discrete state and have a trans transition matrix property')
        self.baseDynamicalSystem = baseDynamicalSystem
        super(DynamicalSystemEnsemble, self).__init__(
            updateOperator=baseDynamicalSystem.trans, updateCls=baseDynamicalSystem.transCls, discrete_time=baseDynamicalSystem.discrete_time)

    def equilibriumState(self):
        """
        Get equilibrium state
        """

        equilibriumDist = super(DynamicalSystemEnsemble, self).equilibriumState()
        equilibriumDist = equilibriumDist / equilibriumDist.sum()

        if np.any(self.updateCls.toDense(equilibriumDist) < 0.0):
            raise Exception("Expect equilibrium state to be positive!")
        return equilibriumDist

    def getUniformDistribution(self):
        return np.ones(self.num_vars) / float(self.num_vars)


class DiscreteStateSystemBase(DynamicalSystemBase):

    """ Has a notion of a transition matrix """

    def __init__(self, num_vars, var_names=None, transCls=None, discrete_time=True, state_dtypes=None):
        super(DiscreteStateSystemBase, self).__init__(num_vars=num_vars,
                                                      var_names=var_names, discrete_time=discrete_time, state_dtypes=state_dtypes)
        self.transCls = transCls if transCls is not None else DEFAULT_TRANSMX_CLASS

    def checkTransitionMatrix(self, trans):
        if trans.shape[0] != trans.shape[1]:
            raise Exception('Expect square transition matrix (got %s)' % trans.shape)

    def stgIgraph(self):
        """
        Return the current state transition graph as an igraph ``Graph`` object.
        """
        import igraph
        return igraph.Graph(zip(*self.trans.nonzero()), directed=True)

    def getAttractorsAndBasins(self):
        """
        Computes the attractors and basins of the current discrete-state dynamical system. 

        :returns: 
        * basinAtts, A list of of list, containing the attractor states for each basin (basin order is from largest basin to smallest).
        * basinStates, A list of of list, containing the attractor states for each basin (basin order is from largest basin to smallest).

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
