"""Module implementing Markov chains.
"""

from __future__ import print_function

import numpy as np
import six
range = six.moves.range
map   = six.moves.map

from . import dynsys
from . import mx

from .utils import hashable_state

TOLERANCE = 1e-10
TRANS_DTYPE = 'float32'

class MarkovChain(dynsys.LinearDynamicalSystem):
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
    base_sys : DiscreteStateDynamicalSystem, optional
        Specifies an underlying dynamical system, possibly vector-valued, to
        whose states the individual outcomes of the Markov chain refer.
    discrete_time : bool, optional
        Whether updating should be done using discrete (default) or continuous
        time dynamics.
    """

    def __init__(self, transition_matrix, base_sys=None, discrete_time=True):
        super(MarkovChain, self).__init__(transition_matrix=transition_matrix,
                                          discrete_time=discrete_time)
        self._check_transition_mx()
        self.base_sys = base_sys
        if base_sys is not None:
            self.state2ndx_map = base_sys.get_state2ndx_map()
            self.ndx2state_map = base_sys.get_ndx2state_map()
        else:
            self.state2ndx_map = None
            self.ndx2state_map = None


    def get_equilibrium_distribution(self):
        """Get equilibrium state (i.e. the stable, equilibrium distribution)
        for this dynamical system.  Uses eigen-decomposition.

        Returns
        -------
        numpy array or scipy.sparse matrix
            Equilibrium distribution
        """

        dist = super(MarkovChain, self).get_equilibrium_distribution()
        dist = dist / dist.sum()

        if np.any(mx.todense(dist) < -TOLERANCE):
            raise Exception("Expect equilibrium state to be positive!")
        return dist

    def get_uniform_distribution(self):
        """Return uniform starting distribution over all system states.
        """
        N = self.transition_matrix.shape[0]
        dist = np.ones(N, TRANS_DTYPE) / N
        return dist

    def _check_transition_mx(self):
        """Internally used function that checks the integrity/format of
        transition matrices.
        """
        trans = self.transition_matrix
        N = trans.shape[0]
        if N != trans.shape[1]:
            raise ValueError('Expect square transition matrix -- got %s:' %
                            (trans.shape,) )

        sums = np.ravel(mx.todense(trans.sum(axis=1)))
        nancount = np.ravel(mx.todense(mx.isnan(trans).sum(axis=1)))

        ok_rows = np.zeros(sums.shape, dtype='bool')
        trg = 1.0 if self.discrete_time else 0.0
        ok_rows[~np.isnan(sums)] = np.abs(sums[~np.isnan(sums)]-trg) < TOLERANCE

        # We allow states to be nan in case their transitions are not defined
        ok_rows[nancount == N] = True

        if not ok_rows.all():
            raise ValueError(
                'For %s-time system trans matrix cols should add to '
                '%d or be all nan (%s)' % 
                ('discrete' if self.discrete_time else 'continuous', trg, sums)
            )

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
        >>> yeast = dynpy.sample_nets.budding_yeast_bn
        >>> bn = dynpy.bn.BooleanNetwork(rules=yeast)
        >>> bnEnsemble = dynpy.markov.MarkovChain.from_deterministic_system(bn, issparse=True)
        >>> init = bnEnsemble.get_uniform_distribution()
        >>> trajectory = bnEnsemble.get_trajectory(init, max_time=80)

        If we wish to project the state of the Markov chain back onto the
        activations of the variables in the underlying system, we can use the
        `ndx2state_mx` matrix. For example:

        >>> import dynpy
        >>> import numpy as np
        >>> yeast = dynpy.sample_nets.budding_yeast_bn
        >>> bn = dynpy.bn.BooleanNetwork(rules=yeast)
        >>> bn_ensemble = dynpy.markov.MarkovChain.from_deterministic_system(bn, issparse=True)
        >>> init = bn_ensemble.get_uniform_distribution()
        >>> final_state = bn_ensemble.iterate(init, max_time=80)
        >>> print(np.ravel(final_state.dot(bn.get_ndx2state_mx())))
        [ 0.          0.05664062  0.07373047  0.07373047  0.92236328  0.          0.
          0.          0.91503906  0.          0.        ]


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

        state2ndx_map = base_sys.get_state2ndx_map()
        translist = [
            (ndx, state2ndx_map[hashable_state(base_sys.iterate(state))], 1.0)
             for ndx, state in six.iteritems(base_sys.get_ndx2state_map()) ]

        nrows, ncols, ndata = zip(*translist)
        mxcls = mx.SparseMatrix if issparse else mx.DenseMatrix
        N = len(state2ndx_map)
        trans = mxcls.from_coords(nrows, ncols, 
                                  np.array(ndata, dtype=TRANS_DTYPE), 
                                  shape=(N,N))
        trans = mx.finalize_mx(trans)

        return cls(transition_matrix=trans, base_sys=base_sys,
            discrete_time=base_sys.discrete_time)

    def project(self, keep_vars, initial_dist=None):
        """Create a Markov chain by projecting an existing 
        Markov chain over a multivariate dynamical system onto a subset of 
        those variables.

        For example:

        >>> import dynpy
        >>> r = [
        ...     ['x1', ['x1','x2'], lambda x1,x2: (x1 and x2) ],
        ...     ['x2', ['x1','x2'], lambda x1,x2: (x1 or  x2) ],
        ... ]
        >>> bn = dynpy.bn.BooleanNetwork(rules=r, mode='FUNCS')
        >>> bnensemble = dynpy.markov.MarkovChain.from_deterministic_system(bn, issparse=False)
        >>> proj = dynpy.markov.MarkovChain.project(bnensemble, [0])
        >>> print(proj.transition_matrix)
        [[ 1.   0. ]
         [ 0.5  0.5]]


        Parameters
        ----------
        keep_vars : list 
            List of variables to keep 
        initial_dist : optional
            Marginalize using this distribution for starting conditions

        """

        if not hasattr(keep_vars, '__iter__'):
            raise ValueError('keep_vars must be list-like')

        if initial_dist is None:
            initial_dist = self.get_uniform_distribution()
        else:
            initial_dist = np.ravel(mx.todense(initial_dist))
            if len([d for d in initial_dist.shape if d > 1]) != 1:
                raise ValueError('initial_dist should be 1-dimensional')
            if not np.isclose(initial_dist.sum(), 1.0):
                raise ValueError('initial_dist should add up to 1')
            #TODO : optimize performance in case that initial_dist is sparse
            # and some starting states have 0 probability

        mxcls = mx.get_cls(self.transition_matrix)

        proj = dynsys.ProjectedStateSpace(
            base_sys=self.base_sys, keep_vars=keep_vars)

        new_s2n = proj.get_state2ndx_map()
        n2s_mx  = self.base_sys.get_ndx2state_mx()

        newN = len(new_s2n)

        rows, cols, data = mxcls.get_coords(self.transition_matrix)

        m_rows = n2s_mx[rows,:][:,keep_vars]
        m_rows_ndxs = [new_s2n[hashable_state(r)] for r in m_rows]

        m_cols = n2s_mx[cols,:][:,keep_vars]
        m_cols_ndxs = [new_s2n[hashable_state(c)] for c in m_cols]

        init_conds = initial_dist[rows]
        trans = mxcls.from_coords(m_rows_ndxs, m_cols_ndxs,
                                  init_conds * data, 
                                  shape=(newN,newN))

        # TODO: write tests for continuous-time projection

        marg_init_prob = np.zeros((trans.shape[0],1))
        for old_ndx, s in enumerate(n2s_mx[:,keep_vars]):
            marg_init_prob[new_s2n[hashable_state(s)]] += initial_dist[old_ndx]

        marg_init_prob[marg_init_prob == 0.] = np.nan
        
        trans = mx.multiplyrows(trans, 1.0/marg_init_prob)

        return MarkovChain(transition_matrix=trans, base_sys=proj,
            discrete_time=self.discrete_time)


class MarkovChainSampler(dynsys.StochasticDynamicalSystem):
    """This class implements a stochastic dynamical system whose trajectory
    represents a sample from a provided Markov chain.

    Parameters
    ----------
    self : :class:`dynpy.markov.MarkovChain`
        Markov chain from which to sample from.  For now, only discrete time 
        is supported.
    """

    def __init__(self, markov_chain):
        if self.discrete_time == False:
            raise Exception('Can only sample from discrete-time MCs')
        self.markov_chain = markov_chain
        super(MarkovChainSampler, self).__init__(
            discrete_time=self.discrete_time)

    def _iterate_1step_discrete(self, start_state):
        mc = self.markov_chain

        if mc.state2ndx_map is not None:
            start_state = mc.state2ndx_map[hashable_state(start_state)]

        probs = mc.transition_matrix[start_state,:]
        probs = np.ravel(mx.get_cls(probs).todense(probs))
        num_states = self.markov_chain.transition_matrix.shape[0]
        r = np.random.choice(np.arange(num_states), None, replace=True, p=probs)

        if mc.ndx2state_map is not None:
            r = mc.ndx2state_map[r]

        return r

    def _iterate_continuous(self, start_state, max_time = 1.0):
        raise NotImplementedError
