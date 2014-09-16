import numpy as np
import scipy.sparse as ss
import dynpy
from dynpy.mx import DenseMatrix, SparseMatrix
from dynpy.graphdynamics import RandomWalker

kc = dynpy.sample_nets.karateclub_net 
initState = np.zeros(kc.shape[0], 'float')
initState[ 5 ] = 1

def veryClose(mx1, mx2):
        fmax = (lambda x: x.max()) if ss.issparse(mx1) else np.max
        fmin = (lambda x: x.min()) if ss.issparse(mx1) else np.min
        return fmax(mx1-mx2) < 1e-5 and fmin(mx1-mx2) > -1e-5

def test_dense_discrete_equil_vs_iter():
        # Dense discrete time
        rw = RandomWalker(graph=kc, transCls=DenseMatrix)
        rwMC = dynpy.dynsys.MarkovChain(rw)

        e1 = rwMC.iterate(initState, max_time = 100)
        e2 = rwMC.equilibriumState()
        assert( veryClose(e1 , e2) )

def test_dense_continuous_vs_discrete():
        # Dense continuous time
        rw = RandomWalker(graph=kc, transCls=DenseMatrix)
        rwMC = dynpy.dynsys.MarkovChain(rw)
        rw = RandomWalker(graph=kc, discrete_time=False, transCls=DenseMatrix)
        rwMCCT = dynpy.dynsys.MarkovChain(rw)
        e2 = rwMC.equilibriumState()
        e2ct = rwMCCT.equilibriumState()
        assert( veryClose(e2ct , e2) )


def test_dense_continuous_equil_vs_iter():
        # Dense continuous time
        rw = RandomWalker(graph=kc, discrete_time=False, transCls=DenseMatrix)
        rwMCCT = dynpy.dynsys.MarkovChain(rw)
        e1 = rwMCCT.iterate(initState, max_time = 100)
        e2ct = rwMCCT.equilibriumState()
        assert( veryClose(e2ct , e1) )


def test_sparse_discrete_equil_vs_iter():
        # Sparse discrete time
        rw = RandomWalker(graph=kc, transCls=SparseMatrix)
        rwMC = dynpy.dynsys.MarkovChain(rw)

        e1 = rwMC.iterate(initState, max_time = 100)
        e2 = rwMC.equilibriumState()
        assert( veryClose(e1 , e2) )

def test_sparse_continuous_vs_discrete():
        # Sparse continuous time
        rw = RandomWalker(graph=kc, transCls=DenseMatrix)
        rwMC = dynpy.dynsys.MarkovChain(rw)
        rw = RandomWalker(graph=kc, discrete_time=False, transCls=SparseMatrix)
        rwMCCT = dynpy.dynsys.MarkovChain(rw)
        e2 = rwMC.equilibriumState()
        e2ct = rwMCCT.equilibriumState()
        assert( veryClose(e2ct , e2) )


def test_sparse_continuous_equil_vs_iter():
        # Sparse continuous time
        rw = RandomWalker(graph=kc, discrete_time=False, transCls=SparseMatrix)
        rwMCCT = dynpy.dynsys.MarkovChain(rw)
        e1 = rwMCCT.iterate(initState, max_time = 100)
        e2ct = rwMCCT.equilibriumState()
        assert( veryClose(e2ct , e1) )

