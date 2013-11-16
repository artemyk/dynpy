import numpy as np
import dynpy


kc_net = dynpy.sample_nets.karateclub_net 
initState = np.zeros(kc_net.shape[0], 'float')
initState[ 5 ] = 1

def veryClose(mx1, mx2):
        return (mx1-mx2).max() < 1e-5 and (mx1-mx2).min() > -1e-5

def test_dense_discrete_equil_vs_iter():
        # Dense discrete time
        rw = dynpy.graphdynamics.RandomWalker(graph=kc_net, transCls = dynpy.mx.DenseMatrix )
        rwMC = dynpy.dynsys.MarkovChain(rw)

        e1 = rwMC.iterate(initState, max_time = 100)
        e2 = rwMC.equilibriumState()
        assert( veryClose(e1 , e2) )

def test_dense_continuous_vs_discrete():
        # Dense continuous time
        rw = dynpy.graphdynamics.RandomWalker(graph=kc_net, transCls = dynpy.mx.DenseMatrix )
        rwMC = dynpy.dynsys.MarkovChain(rw)
        rw = dynpy.graphdynamics.RandomWalker(graph=kc_net, discrete_time = False, transCls = dynpy.mx.DenseMatrix )
        rwMCCT = dynpy.dynsys.MarkovChain(rw)
        e2 = rwMC.equilibriumState()
        e2ct = rwMCCT.equilibriumState()
        assert( veryClose(e2ct , e2) )


def test_dense_continuous_equil_vs_iter():
        # Dense continuous time
        rw = dynpy.graphdynamics.RandomWalker(graph=kc_net, discrete_time = False, transCls = dynpy.mx.DenseMatrix )
        rwMCCT = dynpy.dynsys.MarkovChain(rw)
        e1 = rwMCCT.iterate(initState, max_time = 100)
        e2ct = rwMCCT.equilibriumState()
        assert( veryClose(e2ct , e1) )


def test_sparse_discrete_equil_vs_iter():
        # Sparse discrete time
        rw = dynpy.graphdynamics.RandomWalker(graph=kc_net, transCls = dynpy.mx.SparseMatrix )
        rwMC = dynpy.dynsys.MarkovChain(rw)

        e1 = rwMC.iterate(initState, max_time = 100)
        e2 = rwMC.equilibriumState()
        assert( veryClose(e1 , e2) )

def test_sparse_continuous_vs_discrete():
        # Sparse continuous time
        rw = dynpy.graphdynamics.RandomWalker(graph=kc_net, transCls = dynpy.mx.DenseMatrix )
        rwMC = dynpy.dynsys.MarkovChain(rw)
        rw = dynpy.graphdynamics.RandomWalker(graph=kc_net, discrete_time = False, transCls = dynpy.mx.SparseMatrix )
        rwMCCT = dynpy.dynsys.MarkovChain(rw)
        e2 = rwMC.equilibriumState()
        e2ct = rwMCCT.equilibriumState()
        assert( veryClose(e2ct , e2) )


def test_sparse_continuous_equil_vs_iter():
        # Sparse continuous time
        rw = dynpy.graphdynamics.RandomWalker(graph=kc_net, discrete_time = False, transCls = dynpy.mx.SparseMatrix )
        rwMCCT = dynpy.dynsys.MarkovChain(rw)
        e1 = rwMCCT.iterate(initState, max_time = 100)
        e2ct = rwMCCT.equilibriumState()
        assert( veryClose(e2ct , e1) )

