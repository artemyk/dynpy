import numpy as np
import scipy.sparse as ss
import dynpy
from dynpy.mx import DenseMatrix, SparseMatrix
from dynpy.graphdynamics import RandomWalkerEnsemble

kc = dynpy.sample_nets.karateclub_net
initState = np.zeros(kc.shape[0], 'float')
initState[ 5 ] = 1

def very_close(mx1, mx2):
        fmax = (lambda x: x.max()) if ss.issparse(mx1) else np.max
        fmin = (lambda x: x.min()) if ss.issparse(mx1) else np.min
        return fmax(mx1-mx2) < 1e-5 and fmin(mx1-mx2) > -1e-5

def test_dense_discrete_equil_vs_iter():
        # Dense discrete time
        rw = RandomWalkerEnsemble(graph=kc, issparse=False)

        e1 = rw.iterate(initState, max_time = 100)
        e2 = rw.get_equilibrium_distribution()
        assert( very_close(e1 , e2) )

def test_dense_continuous_vs_discrete():
        # Dense continuous time
        rw1 = RandomWalkerEnsemble(graph=kc, issparse=False)
        rw2 = RandomWalkerEnsemble(graph=kc, discrete_time=False, issparse=False)
        e2 = rw1.get_equilibrium_distribution()
        e2ct = rw2.get_equilibrium_distribution()
        assert( very_close(e2ct , e2) )


def test_dense_continuous_equil_vs_iter():
        # Dense continuous time
        rw = RandomWalkerEnsemble(graph=kc, discrete_time=False, issparse=False)
        e1 = rw.iterate(initState, max_time = 100)
        e2ct = rw.get_equilibrium_distribution()
        assert( very_close(e2ct , e1) )


def test_sparse_discrete_equil_vs_iter():
        # Sparse discrete time
        rw = RandomWalkerEnsemble(graph=kc, issparse=True)

        e1 = rw.iterate(initState, max_time = 100)
        e2 = rw.get_equilibrium_distribution()
        assert( very_close(e1 , e2) )

def test_sparse_continuous_vs_discrete():
        # Sparse continuous time
        rw1 = RandomWalkerEnsemble(graph=kc, issparse=False)
        rw2 = RandomWalkerEnsemble(graph=kc, discrete_time=False, issparse=True)
        e2 = rw1.get_equilibrium_distribution()
        e2ct = rw2.get_equilibrium_distribution()
        assert( very_close(e2ct , e2) )


def test_sparse_continuous_equil_vs_iter():
        # Sparse continuous time
        rw = RandomWalkerEnsemble(graph=kc, discrete_time=False, issparse=True)
        e1 = rw.iterate(initState, max_time = 100)
        e2ct = rw.get_equilibrium_distribution()
        assert( very_close(e2ct , e1) )

