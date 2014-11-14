"""Module implementing some base classes useful for implementing dynamical
systems"""

from __future__ import division, print_function, absolute_import
import six
range = six.moves.range
map   = six.moves.map

import numpy as np

from . import mx
from . import caching
from .utils import hashable_state, readonlydict

# Constants for finding attractors
MAX_ATTRACTOR_LENGTH = 5
TRANSIENT_LENGTH = 30

class DynamicalSystem(object):
    """Base class for dynamical systems.

    Parameters
    ----------
    discrete_time : bool, optional
        Whether updating should be done using discrete (default) or continuous
        time dynamics.
    """

    #: Whether the dynamical system obeys discrete- or continuous-time dynamics
    discrete_time = True

    def __init__(self, discrete_time=True):
        self.discrete_time = discrete_time

        if self.discrete_time:
            self.iterate = self._iterate_discrete
            self.iterate_1step = self._iterate_1step_discrete
        else:
            self.iterate = self._iterate_continuous

    def iterate(self, start_state, max_time):
        """This method runs the dynamical system for `max_time` starting from
        `start_state` and returns the result.  In fact, this method is set at
        run-time by the constructor to either `_iterateDiscrete` or
        `_iterateContinuous` depending on whether the dynamical system object
        is initialized with `discrete_time=True` or `discrete_time=False`. Thus,
        sub-classes should override `_iterateDiscrete` and `_iterateContinuous`
        instead of this method.  See also
        :meth:`dynpy.dynsys.DynamicalSystem.iterateOneStep`

        Parameters
        ----------
        start_state : numpy array or scipy.sparse matrix
            Which state to start from
        max_time : float
            Until which point to run the dynamical system (number of iterations
            for discrete-time systems or time limit for continuous-time systems)

        Returns
        -------
        numpy array or scipy.sparse matrix
            End state
        """
        raise NotImplementedError

    def iterate_1step(self, start_state):
        """
        This method runs a discrete-time dynamical system for 1 timestep. At
        run-time, the construct either repoints this method to `_iterateOneStep`
        for discrete-time systems, or removes it for continuous time systems.

        Parameters
        ----------
        start_state : numpy array or scipy.sparse matrix
            Which state to start from


        Returns
        -------
        numpy array or scipy.sparse matrix
            Iterated state
        """
        raise NotImplementedError

    def _iterate_1step_discrete(self, start_state):
        raise NotImplementedError

    def _iterate_continuous(self, start_state, max_time=1.0):
        raise NotImplementedError

    def _iterate_discrete(self, start_state, max_time=1.0):
        if max_time == 1.0:
            return self.iterate_1step(start_state)
        elif max_time == 0.0:
            return start_state
        else:
            cur_state = start_state
            for i in range(int(max_time)):
                cur_state = self.iterate_1step(cur_state)
            return cur_state

    def get_trajectory(self, start_state, max_time, num_points=None,
                      logscale=False):
        """This method get a trajectory of a dynamical system starting from
        a particular starting state.

        Parameters
        ----------
        start_state : object
            Which state to start from
        max_time : float
            Until which point to run the dynamical system (number of iterations
            for discrete-time systems or time limit for continuous-time systems)
        num_points : int, optional
            How many timepoints to sample the trajectory at.  This determines
            how big each 'step size' is. By default, equal to ``int(max_time)``
        logscale : bool, optional
            Whether to sample the timepoints on a logscale or not (default)

        Returns
        -------
        trajectory: numpy array
            Array of states corresponding to trajectory

        """
        # TODO: would this accumulate error for continuous case?
        if num_points is None:
            num_points = int(max_time)

        if logscale:
            timepoints = np.logspace(0, np.log10(max_time), num=num_points,
                                     endpoint=True, base=10.0)
        else:
            timepoints = np.linspace(0, max_time, num=num_points, endpoint=True)

        cur_state = start_state
        trajectory = [cur_state,]
        start_state_cls = mx.get_cls(start_state)

        for t in range(1, len(timepoints)):
            run_time = timepoints[t]-timepoints[t-1]
            next_state = self.iterate(cur_state, max_time=run_time)
            cur_state = start_state_cls.format_mx(next_state)
            trajectory.append( cur_state )

        return start_state_cls.vstack(trajectory)

class DeterministicDynamicalSystem(DynamicalSystem):
    pass

class StochasticDynamicalSystem(DynamicalSystem):
    pass

class DiscreteStateDynamicalSystem(DynamicalSystem):

    def states(self):
        raise NotImplementedError

    def get_attractor_basins(self, sort=False):
        """Computes the attractors and basins of the current discrete-state
        dynamical system.

        Parameters
        ----------
        sort : bool, optional
            Whether to sort attractors and basin states (slower).

        Returns
        -------
        basin_atts : list of lists
            A list of the the attractor states for each basin (basin order is
            from largest basin to smallest).
        basin_states : list of lists
            A list of all the states in each basin (basin order is from largest
            basin to smallest).

        """

        state_basins = {}
        attractors   = {}

        iteratefunc = self.iterate

        for raw_startstate in self.states():
            startstate = hashable_state(raw_startstate)
            if startstate in state_basins:
                continue

            traj = set()
            cstate = startstate
            while True:                
                traj.add(cstate)
                cstate = hashable_state(iteratefunc(cstate))

                if cstate in traj:  # cycle closed
                    cur_cycle = []
                    cyclestate = cstate 
                    while True:
                        cur_cycle.append(cyclestate)
                        cyclestate = iteratefunc(cyclestate)
                        if cyclestate == cstate:
                            break
                    cur_cycle = tuple(sorted(cur_cycle))
                    if cur_cycle not in attractors:
                        cndx = len(attractors)
                        attractors[cur_cycle] = cndx
                    state_basins[cstate] = attractors[cur_cycle]

                if cstate in state_basins:
                    for s in traj:
                        state_basins[s] = state_basins[cstate]
                    break

        basins = [ [] for _ in range(len(attractors))]
        for state, basin in six.iteritems(state_basins):
            basins[basin].append(state)

        keyfunc = lambda k: (-len(basins[attractors[k]]),k)

        attractor_states = attractors.keys()

        if sort:
            attractor_states = sorted(attractor_states, key=keyfunc)

        basins_states = []
        for att in attractor_states:
            cbasin = basins[attractors[att]]
            if sort:
                cbasin = sorted(cbasin)
            basins_states.append(cbasin)
            
        return attractor_states, basins_states

    def print_attractor_basins(self):
        """Prints the attractors and basin of the dynamical system

        >>> import dynpy
        >>> rules = [ ['a',['a','b'],[1,1,1,0]],['b',['a','b'],[1,0,0,0]]]
        >>> bn = dynpy.bn.BooleanNetwork(rules=rules)
        >>> bn.print_attractor_basins()
        * BASIN 1 : 2 States
        ATTRACTORS:
              a      b
              1      0
        --------------------------------------------------------------------------------
        * BASIN 2 : 1 States
        ATTRACTORS:
              a      b
              0      0
        --------------------------------------------------------------------------------
        * BASIN 3 : 1 States
        ATTRACTORS:
              a      b
              1      1
        --------------------------------------------------------------------------------

        """
        basin_atts, basin_states = self.get_attractor_basins(sort=True)
        for cur_basin_ndx in range(len(basin_atts)):
            print("* BASIN %d : %d States" %
                (cur_basin_ndx+1, len(basin_states[cur_basin_ndx])))
            print("ATTRACTORS:")
            print(self._get_state_row_title())
            for att in basin_atts[cur_basin_ndx]:
                print(self._get_state_row_repr(att))
            print("".join(['-', ] * 80))

    def _get_state_row_repr(self, state):
        return state

    def _get_state_row_title(self):
        return 'State'


class VectorDynamicalSystem(DynamicalSystem):
    """Mix-in for classes implementing dynamics over multivariate systems.

    Parameters
    ----------
    num_vars : int, optional
        How many variables (i.e., how many 'dimensions' or 'nodes' are in the
        dynamical system). Default is 1
    var_names : list, optional
        Names for the variables (optional).  Default is simply the numeric
        indexes of the variables.
    discrete_time : bool, optional
        Whether dynamical system is discrete or continuous time.
    """

    #: The number of variables in the dynamical system
    num_vars = None

    def __init__(self, num_vars, var_names=None, discrete_time=True):
        super(VectorDynamicalSystem,self).__init__(discrete_time)

        self.num_vars = num_vars
        self.var_names = tuple(var_names if var_names is not None
                               else range(self.num_vars))
        """The names of the variables in the dynamical system"""

    # Make this a cached property so its not necessarily run every time a
    # dynamical systems object is created, whether we need it or not
    @caching.cached_data_prop
    def var_name_ndxs(self):
        """A mapping from variables names to their indexes
        """
        return dict((l, ndx) for ndx, l in enumerate(self.var_names))

    def _get_state_row_repr(self, state):
        row_format = "{:>7}" * self.num_vars
        return row_format.format(*state)

    def _get_state_row_title(self):
        row_format = "{:>7}" * self.num_vars
        return row_format.format(*self.var_names)

class DiscreteStateVectorDynamicalSystem(VectorDynamicalSystem,
    DiscreteStateDynamicalSystem):

    @caching.cached_data_method
    def get_ndx2state_mx(self):
        #: ``(num_states, num_vars)``-shaped matrix that maps from state indexes
        #: to representations in terms of activations of the variables.
        return np.vstack(self.states())

    @caching.cached_data_method
    def get_ndx2state_map(self):
        return readonlydict( enumerate(self.states()) )

    @caching.cached_data_method
    def get_state2ndx_map(self):
        return readonlydict( 
            { hashable_state(v):k 
              for k, v in six.iteritems(self.get_ndx2state_map()) })


class ProjectedStateSpace(DiscreteStateVectorDynamicalSystem):
    # TODO: Document
    def __init__(self, keep_vars, base_sys, *kargs, **kwargs):
        self.keep_vars = keep_vars
        self.base_sys = base_sys

        var_names = None
        if base_sys.var_names is not None:
            var_names = [base_sys.var_names[i] for i in keep_vars]

        super(ProjectedStateSpace, self).__init__(
            num_vars=len(keep_vars),
            var_names=var_names,
            *kargs, **kwargs)

    def states(self):
        done = set()
        n2smx = self.base_sys.get_ndx2state_mx()
        for c in map(hashable_state, n2smx[:,self.keep_vars]):
            if c not in done:
                done.add(c)
                yield c


class LinearDynamicalSystem(VectorDynamicalSystem):
    # TODO: TESTS
    """This class implements linear dynamical systems, whether continuous or
    discrete-time.  It is also used by :class:`dynpy.dynsys.MarkovChain` to
    implement Markov Chain (discrete-case) or  master equation (continuous-case)
    dynamics.

    For attribute definitions, see documentation of
    :class:`dynpy.dynsys.DynamicalSystem`.

    Parameters
    ----------
    transition_matrix : numpy array or scipy.sparse matrix
        Matrix defining the evolution of the dynamical system, i.e. the
        :math:`\\mathbf{A}` in
        :math:`\\mathbf{x_{t+1}} = \\mathbf{x_{t}}\\mathbf{A}` (in the
        discrete-time case) or
        :math:`\\dot{\\mathbf{x}} = \\mathbf{x}\\mathbf{A}` (in the
        continuous-time case)
    discrete_time : bool, optional
        Whether updating should be done using discrete (default) or continuous
        time dynamics.
    """

    #: Transition matrix for linear system.
    transition_matrix = None

    def __init__(self, transition_matrix, discrete_time=True):
        super(LinearDynamicalSystem, self).__init__(
            num_vars=transition_matrix.shape[0], 
            discrete_time=discrete_time)
        self.transition_matrix = transition_matrix
        self.stable_eigenvalue = 1.0 if discrete_time else 0.0

    def equilibrium_distribution(self):
        """Get equilibrium state of dynamical system using eigen-decomposition

        Returns
        -------
        numpy array or scipy.sparse matrix
            Equilibrium state
        """

        vals, vecs = mx.get_largest_left_eigs(self.transition_matrix)
        equil_evals = np.flatnonzero(np.abs(vals-self.stable_eigenvalue) < 1e-8)
        if len(equil_evals) != 1:
            raise Exception("Expected one stable eigenvalue, but found " +
                            "%d instead (%s)" % (len(equil_evals), equil_evals))

        dist = np.real_if_close(np.ravel(vecs[equil_evals, :]))
        if np.any(np.iscomplex(equil_evals)):
            raise Exception("Expect equilibrium state to be real! %s" %
                            equil_evals)

        return mx.format_mx(dist)

    def _iterate_1step_discrete(self, start_state):
        # For discrete time systems, one step
        r = mx.format_mx(start_state).dot(self.transition_matrix)
        return r

    def _iterate_discrete(self, start_state, max_time=1.0):
        # For discrete time systems
        cls = mx.get_cls(self.transition_matrix)
        r = cls.format_mx(start_state).dot(
                cls.pow(self.transition_matrix, int(max_time)))
        return r

    def _iterate_continuous(self, start_state, max_time=1.0):
        cls = mx.get_cls(self.transition_matrix)
        curStartStates = cls.format_mx(start_state)
        r = curStartStates.dot(
              cls.expm(max_time * (self.transition_matrix)))
        return r

    # TODO
    # def getMultistepDynsys(self, num_iters):
    #     import copy
    #     rObj = copy.copy(self)
    #     rObj.trans = self.transition_matrixCls.pow(self.transition_matrix, num_iters)
    #     return rObj



