# Also look at https://wiki.python.org/moin/PyDSTool


"""
TODO:
- Fix sparse random walker
- Fix mapping from ids to states

- Continuous time
- Marginalization

- Test 'get trajectory' (or for list of timepoints, etc.)


- Perturbation modelling?
- Ensemble of BNs? (i.e. RBNs?)  Use slots perhaps, to minimize memory overhead
- Santosh's algebraic model?

- Document!


"""

import dynpy.bn
import dynpy.dynsys
import dynpy.sample_bn_nets
import dynpy.mxutils
#from dynpy import dynsys, bn
import numpy as np



import dynpy.ca



import networkx as nx
import matplotlib.pyplot as plt
import dynpy.graphdynamics

num_steps = 50
rw = dynpy.graphdynamics.RandomWalker(graph=nx.to_numpy_matrix( nx.karate_club_graph() ) )
spacetime = np.zeros(shape=(num_steps,rw.num_nodes), dtype='int')

cState = np.zeros(rw.num_nodes, 'int')
cState[ 3 ] = 1.0

for i in range(num_steps):
    spacetime[i,:] = cState
    cState = rw.iterateState(cState)



rwens = dynpy.dynsys.DynamicalSystemEnsemble(rw)
print rwens.iterateState(startState = rwens.getUniformDistribution())



import networkx as nx, matplotlib.pyplot as plt
import dynpy.graphdynamics

num_steps = 100

import networkx as nx, matplotlib.pyplot as plt
import dynpy.graphdynamics

num_steps = 10
rw = dynpy.graphdynamics.RandomWalker(graph=nx.to_numpy_matrix( nx.karate_club_graph() ), transMatrixClass=dynpy.mx.DenseMatrix )
rwEnsemble = dynpy.dynsys.DynamicalSystemEnsemble(rw)
spacetime = np.zeros(shape=(num_steps,rw.num_nodes))

cState = np.zeros(rw.num_nodes)
cState[ 5 ] = 1

for i in range(num_steps):
    spacetime[i,:] = cState
    cState = rwEnsemble.iterateState(cState)

print spacetime
asdfasf

# a = dynpy.bn.MultivariateSys(num_nodes=2, trans=np.eye(2))

#        multistepTrans = mxutils.raise_matrix_power(self.trans, TRANSIENT_LENGTH)

# print dynpy.sample_bn_nets.yeast
b = dynpy.bn.BooleanNetworkFromTruthTables(rules=dynpy.sample_bn_nets.yeast)
#b = dynpy.bn.BooleanNetworkFromTruthTables(rules=[['A',['A','B'],lambda A,B: A and B],['B',['A','B'],lambda A,B: A or B],])
print b.trans
b = dynpy.bn.BooleanNetworkFromFuncs(rules=[['A',['A','B'],lambda A,B: A and B],['B',['A','B'],lambda A,B: A or B],])
print b.trans

REPORTED_YEAST_NET_ATTRACTOR_SIZES = [1764, 151, 109, 9, 7, 7, 1]

b = dynpy.bn.BooleanNetworkFromTruthTables(rules=dynpy.sample_bn_nets.yeast)
atts, attbasins = b.getAttractorsAndBasins()
print map(len, attbasins)
print map(len, attbasins) == REPORTED_YEAST_NET_ATTRACTOR_SIZES
del b

#print dynpy.mxutils.raise_matrix_power(b.dyn.trans, 10)
#asdsf
# print b.dyn.trans

asdfa

print b.printAttractorsAndBasins()
asdf
print b.dyn.stg_igraph().components().__dict__
asdfdsa

asdf
print b.dyn.stg_igraph().components()

print b.dyn.getAttractorDistribution()
# print b.dyn.equilibriumDistribution()
b = dynpy.bn.BooleanNetworkFromRules(rules=dynpy.sample_bn_nets.yeast, dynsys_class=dynpy.dynsys.StochasticDynSys)

# TODO:
# - Random walker dynamical system
# - Implement laplacian, normalized laplacian of state-transition graph
# - 'Mapped network'
# - Starting state distribution code
# - Marginalization code

# Tests to do:
# - Init simple dynamical system, check attractors, both sparse, non-sparse, mapped
# - Do simple AND 2-node net, check attractors and transition graph
# - Marginalize simple AND 2-node net, check transition graph for different types (sparse/non-sparse/mapped)
# - 
# - Load yeast network, get attractors correctly
# - Marginalize, check that everything makes sense 
# - Check random walker from karate code
