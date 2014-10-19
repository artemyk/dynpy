from __future__ import print_function

import numpy as np
import six
range = six.moves.range

from . import dynsys
from . import mx

from . import caching

class MarkovChain(dynsys.LinearSystem):

    # POSSIBILITY FOR CONFUSION!!!!
    # MarkovChain can be understood as deterministic, vector-valued dynamical 
    # system each of whose 'states' is a probability distribution.
    # Alternatively, it can be seen as stochastic vector-or-non-vector valued 
    # dynamical system.  
    # Here it referes to the first sense.

    # TODO FIX DOCUMENTATION

    """This is a base class for discrete-state dynamical systems.  It provides
    a transition matrix indicating transitions between system states.

    Parameters
    ----------
    updateOperator : numpy array or scipy.sparse matrix
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

    @caching.cached_data_prop
    def ndx2stateMx(self):
        #: ``(num_states, num_vars)``-shaped matrix that maps from state indexes
        #: to representations in terms of activations of the variables.
        num_vars = len(next(six.iterkeys(self.state2ndxMap)))

        mx = np.zeros(shape=(len(self.state2ndxMap),num_vars))
        for state, ndx in six.iteritems(self.state2ndxMap):
            mx[ndx,:] = state
            
        return mx

    def __init__(self, updateOperator, state2ndxMap=None, discrete_time=True):
        # !!! self.base_dynsys = base_dynsys
        super(MarkovChain, self).__init__(updateOperator=updateOperator,
                                          discrete_time=discrete_time)
        self.checkTransitionMatrix(updateOperator)

        if state2ndxMap is None:
            self.state2ndxMap = None
            self.ndx2stateMap = None
            self.ndx2state = lambda x: x
            self.state2ndx = lambda x: x
        else:
            self.state2ndxMap = state2ndxMap
            self.ndx2stateMap = dict((v,k) for k,v in six.iteritems(state2ndxMap))
            self.ndx2state = lambda x: state2ndxMap[x]
            self.state2ndx = lambda x: ndx2stateMap[x]

    def equilibriumState(self):
        """Get equilibrium state (i.e. the stable, equilibrium distribution)
        for this dynamical system.  Uses eigen-decomposition.

        Returns
        -------
        numpy array or scipy.sparse matrix
            Equilibrium distribution
        """

        equilibriumDist = super(MarkovChain, self).equilibriumState()
        equilibriumDist = equilibriumDist / equilibriumDist.sum()

        if np.any(mx.todense(equilibriumDist) < 0.0):
            raise Exception("Expect equilibrium state to be positive!")
        return equilibriumDist

    def getUniformDistribution(self):
        """Gets uniform starting distribution over all system states.
        """
        N = self.updateOperator.shape[0]
        return np.ones(N) / float(N)

    def checkTransitionMatrix(self, trans):
        """Internally used function that checks the integrity/format of
        transition matrices.
        """
        if trans.shape[0] != trans.shape[1]:
            raise Exception('Expect square transition matrix (got %s)' %
                            trans.shape)
        sums = mx.todense(trans.sum(axis=1))
        if self.discrete_time:
            if not np.allclose(sums, 1.0):
                raise Exception('For discrete system, state transitions ' +
                                'entries should add up to 1.0 (%s)' % sums)
        else:
            if not np.allclose(sums, 0.0):
                raise Exception('For continuous system, state transitions ' +
                                'entries should add up to 0.0 (%s)' % sums)


    """
    Alternative constructor that creates a Markov chain from the transitions
    of a deterministic system.


    !!! Parameters
    ----------
    issparse : bool, optional
        Whether transition matrix should be in sparse or dense matrix format


    >>> r = [['x1', ['x1','x2'], lambda x1,x2: (x1 and x2) ],
    ...      ['x2', ['x1','x2'], lambda x1,x2: (x1 or  x2) ]]
    >>> bn = dynpy.bn.BooleanNetwork(rules=r, mode='FUNCS')
    >>> mc = dynpy.markov.MarkovChain.from_deterministic_system(bn)    
    array([[ 1.,  0.,  0.,  0.],
           [ 0.,  1.,  0.,  0.],
           [ 0.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  1.]])

   """
    @classmethod
    def from_deterministic_system(cls, base_sys, issparse=False):
        # ALSO MUST BE DiscreteStateDynamicalSystem TODO, not just deterministic
        """This alternative constructor creates a a Markov Chain from the
        transitions of an underlying deterministic system.
        It maintains properties of the underlying system, such as the
        sparsity of the state transition matrix, and whether the system is discrete
        or continuous-time.  The underlying system must be an instance of
        :class:`dynpy.dynsys.DiscreteStateSystemBase` and provide a transition
        matrix in the form of a `trans` property.

        For example, for a Boolean network:

        >>> import dynpy
        >>> yeast = dynpy.sample_nets.yeast_cellcycle_bn
        >>> bn = dynpy.bn.BooleanNetwork(rules=yeast)
        >>> bnEnsemble = dynpy.markov.MarkovChain.from_deterministic_system(bn, issparse=True)
        >>> init = bnEnsemble.getUniformDistribution()
        >>> trajectory = bnEnsemble.getTrajectory(init, max_time=80)

        If we wish to project the state of the Markov chain back onto the
        activations of the variables in the underlying system, we can use the
        `ndx2stateMx` matrix. For example:

        >>> import dynpy
        >>> import numpy as np
        >>> yeast = dynpy.sample_nets.yeast_cellcycle_bn
        >>> bn = dynpy.bn.BooleanNetwork(rules=yeast)
        >>> bnEnsemble = dynpy.markov.MarkovChain.from_deterministic_system(bn, issparse=True)
        >>> init = bnEnsemble.getUniformDistribution()
        >>> final_state = bnEnsemble.iterate(init, max_time=80)
        >>> print(np.ravel(final_state.dot(bnEnsemble.ndx2stateMx)))
        [ 0.          0.05664062  0.07373047  0.07373047  0.91503906  0.          0.
          0.          0.92236328  0.          0.        ]


        Parameters
        ----------
        base_sys : :class:`dynpy.dynsys.DeterministicDynamicalSystem`
            an object containing the underlying dynamical system over which an
            ensemble will be created.

        """

        # TODO issparse
        #self.base_dynsys = base_dynsys

        if not isinstance(base_sys, dynsys.DeterministicDynamicalSystem):
            raise ValueError('dynsys should be instance of '
                             'DeterministicDynamicalSystem')

        if not base_sys.discrete_time:
            raise ValueError('dynsys should be a discrete-time system')

        state2ndxMap = dict( (state, ndx)
                             for ndx, state in enumerate(base_sys.states()))

        N = len(state2ndxMap)

        mxcls = mx.SparseMatrix if issparse else mx.DenseMatrix
        trans = mxcls.createEditableZerosMx(shape=(N, N))

        #for statendx, nextstatendx in base_sys.ndx_trans():
        #    trans[statendx, nextstatendx] = 1.
        for state in base_sys.states():
            nextstate = base_sys.iterate(state)
            trans[state2ndxMap[state], state2ndxMap[nextstate]] = 1.

        trans = mx.finalizeMx(trans)

        return cls(updateOperator=trans, state2ndxMap=state2ndxMap,
            discrete_time=base_sys.discrete_time)


    @classmethod
    def marginalize(cls, markov_chain, keep_vars, initial_dist=None):
        # TODO doc, alternative constructor
        # TODO --- test that base markov chain is multivariate
        """
        Example:

        >>> import dynpy
        >>> r = [
        ...     ['x1', ['x1','x2'], lambda x1,x2: (x1 and x2) ],
        ...     ['x2', ['x1','x2'], lambda x1,x2: (x1 or  x2) ],
        ... ]
        >>> bn = dynpy.bn.BooleanNetwork(rules=r, mode='FUNCS')
        >>> bnensemble = dynpy.markov.MarkovChain.from_deterministic_system(bn)
        >>> marg = dynpy.markov.MarkovChain.marginalize(bnensemble, [0])
        >>> print(marg.updateOperator)
        [[ 1.   0. ]
         [ 0.5  0.5]]

        """
        def marginalize_state(state):
            return dynsys.VectorDynamicalSystem.vector_state_class(
                [state[i] for i in keep_vars])

        def states():
            done = set()
            for fullstate in markov_chain.state2ndxMap:
                c = marginalize_state(fullstate)
                if c not in done:
                    done.add(c)
                    yield c

        state2ndxMap = dict( (state, ndx)
                             for ndx, state in enumerate(states()))

        N = len(state2ndxMap)

        if initial_dist is None:
            initial_dist = markov_chain.getUniformDistribution()

        mxcls = mx.get_cls(markov_chain.updateOperator)
        trans = mxcls.createEditableZerosMx(shape=(N, N))
        for i, sstate in six.iteritems(markov_chain.ndx2stateMap):
            initial_p = initial_dist[i]
            mI = state2ndxMap[marginalize_state(sstate)]
            for j, estate in six.iteritems(markov_chain.ndx2stateMap):
                mJ = state2ndxMap[marginalize_state(estate)]
                trans[mI, mJ] += initial_p * markov_chain.updateOperator[i,j]

        trans = trans/trans.sum(axis=1)[:,np.newaxis]
        trans = mxcls.finalizeMx(trans)

        return cls(updateOperator=trans, state2ndxMap=state2ndxMap)



class MarkovChainSampler(dynsys.StochasticDynamicalSystem):
    # TODO document
    # TODO change in tutorial example with random walker

    def __init__(self, markov_chain):
        if markov_chain.discrete_time == False:
            raise Exception('Can only sample from discrete-time MCs')
        self.markov_chain = markov_chain
        super(MarkovChainSampler, self).__init__(discrete_time=True)

    def _iterateOneStepDiscrete(self, startState):
        mc = self.markov_chain
        probs = mc.updateOperator[mc.state2ndx(startState),:]
        probs = np.ravel(mx.get_cls(probs).todense(probs))
        num_states = self.markov_chain.updateOperator.shape[0]
        r = np.random.choice(np.arange(num_states), None, replace=True, p=probs)
        return mc.ndx2state(r)

    def _iterateContinuous(self, startState, max_time = 1.0):
        raise NotImplementedError
