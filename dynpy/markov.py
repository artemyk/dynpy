"""Module implementing Markov chains.
"""

from __future__ import print_function

import numpy as np
import six
range = six.moves.range

from . import dynsys
from . import mx

from . import caching

class MarkovChain(dynsys.LinearSystem):
    """This class implements Markov chains.

    There is some potential for confusion regarding the term 'Markov chain'. It 
    may be used to indicate a stochastic dynamical system, which transitions 
    from state to state with different probabilities.  Alternatively, and in the 
    sense used in `dynpy`, a Markov chain refers to a deterministic, 
    multivariate dynamical that transforms probability distributions over some 
    underlying to distributions into other probability distributions.

    Parameters
    ----------
    transition_matrix : numpy array or scipy.sparse matrix
        Matrix defining the evolution of the dynamical system, i.e. the
        :math:`\\mathbf{A}` in
        :math:`\\mathbf{x_{t+1}} = \\mathbf{x_{t}}\\mathbf{A}` (in the
        discrete-time case) or
        :math:`\\dot{\\mathbf{x}} = \\mathbf{x}\\mathbf{A}` (in the
        continuous-time case)
    state2ndx_map : dict, optional
        Often a Markov chain will be defined over the states of another 
        underlying dynamical system, possibly vector-valued.  This dictionary
        maps from integer-valued states used by the Markov chain to another,
        possibly multivariate state-space.
    discrete_time : bool, optional
        Whether updating should be done using discrete (default) or continuous
        time dynamics.
    """

    def __init__(self, transition_matrix, state2ndx_map=None, discrete_time=True):
        super(MarkovChain, self).__init__(transition_matrix=transition_matrix,
                                          discrete_time=discrete_time)
        self._check_transition_mx()

        if state2ndx_map is None:
            self.state2ndx_map = None
            self.ndx2state_map = None
            self.state2ndx = lambda x: x
            self.ndx2state = lambda x: x
        else:
            self.state2ndx_map = state2ndx_map
            self.ndx2state_map = dict((v,k) for k,v in six.iteritems(state2ndx_map))
            self.state2ndx = lambda x: self.state2ndx_map[dynsys.hashable_state(x)]
            self.ndx2state = lambda x: self.ndx2state_map[dynsys.hashable_state(x)]

    @caching.cached_data_prop
    def ndx2state_mx(self):
        #: ``(num_states, num_vars)``-shaped matrix that maps from state indexes
        #: to representations in terms of activations of the variables.

        if self.state2ndx_map is None:
            return np.eye(self.transition_matrix.shape[0])

        else:
            num_vars = len(next(six.iterkeys(self.state2ndx_map)))

            mx = np.zeros(shape=(len(self.state2ndx_map),num_vars))
            for state, ndx in six.iteritems(self.state2ndx_map):
                mx[ndx,:] = state
                
            return mx

    def equilibrium_distribution(self):
        """Get equilibrium state (i.e. the stable, equilibrium distribution)
        for this dynamical system.  Uses eigen-decomposition.

        Returns
        -------
        numpy array or scipy.sparse matrix
            Equilibrium distribution
        """

        dist = super(MarkovChain, self).equilibrium_distribution()
        dist = dist / dist.sum()

        if np.any(mx.todense(dist) < 0.0):
            raise Exception("Expect equilibrium state to be positive!")
        return dist

    def get_uniform_distribution(self):
        """Return uniform starting distribution over all system states.
        """
        N = self.transition_matrix.shape[0]
        return np.ones(N) / float(N)

    def _check_transition_mx(self):
        """Internally used function that checks the integrity/format of
        transition matrices.
        """
        if self.transition_matrix.shape[0] != self.transition_matrix.shape[1]:
            raise ValueError('Expect square transition matrix -- got %s:' %
                            (self.transition_matrix.shape,) )
        sums = mx.todense(self.transition_matrix.sum(axis=1))
        if self.discrete_time:
            if not np.allclose(sums, 1.0):
                raise ValueError('For discrete system, state transitions ' +
                                'entries should add up to 1.0 (%s)' % sums)
        else:
            if not np.allclose(sums, 0.0):
                raise ValueError('For continuous system, state transitions ' +
                                'entries should add up to 0.0 (%s)' % sums)


    @classmethod
    def from_deterministic_system(cls, base_sys, issparse=True):
        """Alternative constructor creates a a Markov Chain from the transitions
        of an underlying deterministic system. It maintains properties of the 
        underlying system, such as the sparsity of the state transition matrix,
        and whether the system is discrete or continuous-time.  The underlying 
        system must be an instance of
        :class:`dynpy.dynsys.DeterministicDynamicalSystem` and
        :class:`dynpy.dynsys.DiscreteStateDynamicalSystem`.

        For example, for a Boolean network:

        >>> import dynpy
        >>> yeast = dynpy.sample_nets.yeast_cellcycle_bn
        >>> bn = dynpy.bn.BooleanNetwork(rules=yeast)
        >>> bnEnsemble = dynpy.markov.MarkovChain.from_deterministic_system(bn, issparse=True)
        >>> init = bnEnsemble.get_uniform_distribution()
        >>> trajectory = bnEnsemble.get_trajectory(init, max_time=80)

        If we wish to project the state of the Markov chain back onto the
        activations of the variables in the underlying system, we can use the
        `ndx2state_mx` matrix. For example:

        >>> import dynpy
        >>> import numpy as np
        >>> yeast = dynpy.sample_nets.yeast_cellcycle_bn
        >>> bn = dynpy.bn.BooleanNetwork(rules=yeast)
        >>> bn_ensemble = dynpy.markov.MarkovChain.from_deterministic_system(bn, issparse=True)
        >>> init = bn_ensemble.get_uniform_distribution()
        >>> final_state = bn_ensemble.iterate(init, max_time=80)
        >>> print(np.ravel(final_state.dot(bn_ensemble.ndx2state_mx)))
        [ 0.          0.05664062  0.07373047  0.07373047  0.91503906  0.          0.
          0.          0.92236328  0.          0.        ]


        Parameters
        ----------
        base_sys : object
            Dynamical system over whose states the Markov chain will be defined
        issparse : bool, optional
            Whether transition matrix should be in sparse or dense matrix format

        """

        if not isinstance(base_sys, dynsys.DeterministicDynamicalSystem):
            raise ValueError('dynsys should be instance of '
                             'DeterministicDynamicalSystem')
        if not isinstance(base_sys, dynsys.DiscreteStateDynamicalSystem):
            raise ValueError('dynsys should be instance of '
                             'DiscreteStateDynamicalSystem')

        if not base_sys.discrete_time:
            raise ValueError('dynsys should be a discrete-time system')

        state2ndx_map = dict( (state, ndx)
                             for ndx, state in enumerate(base_sys.states()))

        N = len(state2ndx_map)

        mxcls = mx.SparseMatrix if issparse else mx.DenseMatrix
        trans = mxcls.create_editable_zeros_mx(shape=(N, N))

        for state in base_sys.states():
            nextstate = base_sys.iterate(state)
            trans[state2ndx_map[state], state2ndx_map[nextstate]] = 1.

        trans = mx.finalize_mx(trans)

        return cls(transition_matrix=trans, state2ndx_map=state2ndx_map,
            discrete_time=base_sys.discrete_time)


    @classmethod
    def marginalize(cls, markov_chain, keep_vars, initial_dist=None):
        """Alternative constructor that creates a Markov chain by marginalizing
        a Markov chain over a multivariate dynamical system onto a subset of 
        those variables.

        For example:

        >>> import dynpy
        >>> r = [
        ...     ['x1', ['x1','x2'], lambda x1,x2: (x1 and x2) ],
        ...     ['x2', ['x1','x2'], lambda x1,x2: (x1 or  x2) ],
        ... ]
        >>> bn = dynpy.bn.BooleanNetwork(rules=r, mode='FUNCS')
        >>> bnensemble = dynpy.markov.MarkovChain.from_deterministic_system(bn, issparse=False)
        >>> marg = dynpy.markov.MarkovChain.marginalize(bnensemble, [0])
        >>> print(marg.transition_matrix)
        [[ 1.   0. ]
         [ 0.5  0.5]]


        Parameters
        ----------
        markov_chain : class:`dynpy.markov.MarkovChain`
            Markov chain to marginalize
        keep_vars : list 
            List of variables to keep 
        initial_dist : optional
            Marginalize using this distribution for starting conditions

        """

        def marginalize_state(state):
            return dynsys.hashable_state(np.array(
                [state[i] for i in keep_vars]))

        def states():
            done = set()
            for fullstate in markov_chain.state2ndx_map:
                c = marginalize_state(fullstate)
                if c not in done:
                    done.add(c)
                    yield c

        state2ndx_map = dict( (state, ndx)
                             for ndx, state in enumerate(states()))

        N = len(state2ndx_map)

        if initial_dist is None:
            initial_dist = markov_chain.get_uniform_distribution()

        mxcls = mx.get_cls(markov_chain.transition_matrix)
        trans = mxcls.create_editable_zeros_mx(shape=(N, N))
        for i, sstate in six.iteritems(markov_chain.ndx2state_map):
            initial_p = initial_dist[i]
            mI = state2ndx_map[marginalize_state(sstate)]
            for j, estate in six.iteritems(markov_chain.ndx2state_map):
                mJ = state2ndx_map[marginalize_state(estate)]
                trans[mI, mJ] += initial_p * markov_chain.transition_matrix[i,j]

        trans = mxcls.multiply(trans, 1.0/trans.sum(axis=1))
        trans = mxcls.finalize_mx(trans)

        return cls(transition_matrix=trans, state2ndx_map=state2ndx_map)



class MarkovChainSampler(dynsys.StochasticDynamicalSystem):
    """This class implements a stochastic dynamical system whose trajectory
    represents a sample from a provided Markov chain.

    Parameters
    ----------
    markov_chain : :class:`dynpy.markov.MarkovChain`
        Markov chain from which to sample from.  For now, only discrete time 
        is supported.
    """

    def __init__(self, markov_chain):
        if markov_chain.discrete_time == False:
            raise Exception('Can only sample from discrete-time MCs')
        self.markov_chain = markov_chain
        super(MarkovChainSampler, self).__init__(
            discrete_time=markov_chain.discrete_time)

    def _iterate_1step_discrete(self, start_state):
        mc = self.markov_chain
        probs = mc.transition_matrix[mc.state2ndx(start_state),:]
        probs = np.ravel(mx.get_cls(probs).todense(probs))
        num_states = self.markov_chain.transition_matrix.shape[0]
        r = np.random.choice(np.arange(num_states), None, replace=True, p=probs)
        return mc.ndx2state(r)

    def _iterate_continuous(self, start_state, max_time = 1.0):
        raise NotImplementedError
