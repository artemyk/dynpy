# Also look at https://wiki.python.org/moin/PyDSTool


"""
TODO:
- Marginalization


- Perturbation modelling?
- Ensemble of BNs? (i.e. RBNs?)  Use slots perhaps, to minimize memory overhead
- Santosh's algebraic model?

- Document!


"""

# TODO:
# - Implement laplacian, normalized laplacian of state-transition graph
# - Marginalization code

# Tests to do:
# - Init simple dynamical system, check attractors, both sparse, non-sparse, mapped
# - Do simple AND 2-node net, check attractors and transition graph
# - Marginalize simple AND 2-node net, check transition graph for different types (sparse/non-sparse/mapped)
# - 
# - Load yeast network, get attractors correctly
# - Marginalize, check that everything makes sense 
# - Check random walker from karate code


import dynpy
import numpy as np


#bn = dynpy.bn.BooleanNetwork(rules=dynpy.sample_bn_nets.yeast)
#bnEnsemble = dynpy.dynsys.DynamicalSystemEnsemble(bn)
#_  = bnEnsemble.getTrajectory(bnEnsemble.getUniformDistribution(), last_timepoint=80).dot(bn.state2ndxMx)




rw = dynpy.graphdynamics.RandomWalker(graph=dynpy.sample_nets.karateclub_net, discrete_time=False, transCls = dynpy.mx.SparseMatrix )
rwEnsemble = dynpy.dynsys.DynamicalSystemEnsemble(rw)


initState = np.zeros(rw.num_vars, 'float')
initState[ 5 ] = 1



e1 = rwEnsemble.iterate(initState, max_time = 100)
print e1

e2 = rwEnsemble.equilibriumState()


print e2

initState = np.zeros(34, 'float')
initState[ 5 ] = 1


rw = dynpy.graphdynamics.RandomWalker(graph=dynpy.sample_nets.karateclub_net)

initState = np.zeros(rw.num_vars, 'int')
initState[ 5 ] = 1

print rw.getTrajectory(initState, last_timepoint=80)



#rw = dynpy.graphdynamics.RandomWalker(graph=nx.to_numpy_matrix( nx.karate_club_graph() ), discrete_time = False, transCls=dynpy.mx.SparseMatrix )
#rwEnsemble = dynpy.dynsys.DynamicalSystemEnsemble(rw)
#rwEnsemble.getTrajectory(initState, last_timepoint=10, logscale=True)

rw = dynpy.graphdynamics.RandomWalker(graph=dynpy.sample_nets.karateclub_net, transCls=dynpy.mx.SparseMatrix )
rwEnsemble = dynpy.dynsys.DynamicalSystemEnsemble(rw)
rwEnsemble.getTrajectory(initState, last_timepoint=10, logscale=True)


rw = dynpy.graphdynamics.RandomWalker(graph=dynpy.sample_nets.karateclub_net, transCls=dynpy.mx.DenseMatrix )
rwEnsemble = dynpy.dynsys.DynamicalSystemEnsemble(rw)
rwEnsemble.getTrajectory(initState, last_timepoint=80, logscale=True)





b = dynpy.bn.BooleanNetwork(rules=dynpy.sample_nets.yeast_cellcycle_bn)
#b = dynpy.bn.BooleanNetworkFromTruthTables(rules=[['A',['A','B'],lambda A,B: A and B],['B',['A','B'],lambda A,B: A or B],])

b = dynpy.bn.BooleanNetwork(mode='FUNCS',rules=[['A',['A','B'],lambda A,B: A and B],['B',['A','B'],lambda A,B: A or B],])
print b.trans

REPORTED_YEAST_NET_ATTRACTOR_SIZES = [1764, 151, 109, 9, 7, 7, 1]

b = dynpy.bn.BooleanNetwork(rules=dynpy.sample_nets.yeast_cellcycle_bn)
atts, attbasins = b.getAttractorsAndBasins()
print map(len, attbasins)
print map(len, attbasins) == REPORTED_YEAST_NET_ATTRACTOR_SIZES
del b


import dynpy.ca



import matplotlib.pyplot as plt
import dynpy.graphdynamics

num_steps = 50
rw = dynpy.graphdynamics.RandomWalker(graph=dynpy.sample_nets.karateclub_net)
spacetime = np.zeros(shape=(num_steps,rw.num_vars), dtype='int')

cState = np.zeros(rw.num_vars, 'int')
cState[ 3 ] = 1.0

for i in range(num_steps):
    spacetime[i,:] = cState
    cState = rw.iterate(cState)



rwens = dynpy.dynsys.DynamicalSystemEnsemble(rw)
print rwens.iterateOneStep(startState = rwens.getUniformDistribution())



import networkx as nx, matplotlib.pyplot as plt
import dynpy.graphdynamics

num_steps = 100

import networkx as nx, matplotlib.pyplot as plt
import dynpy.graphdynamics

num_steps = 10
rw = dynpy.graphdynamics.RandomWalker(graph=dynpy.sample_nets.karateclub_net, discrete_time=False, transCls=dynpy.mx.DenseMatrix )
rwEnsemble = dynpy.dynsys.DynamicalSystemEnsemble(rw)
spacetime = np.zeros(shape=(num_steps,rw.num_vars))

cState = np.zeros(rw.num_vars)
cState[ 5 ] = 1

for i in range(num_steps):
    spacetime[i,:] = cState
    cState = rwEnsemble.iterate(cState)

print spacetime

# a = dynpy.bn.MultivariateSys(num_vars=2, trans=np.eye(2))

#        multistepTrans = mxutils.raise_matrix_power(self.trans, TRANSIENT_LENGTH)

# print dynpy.sample_bn_nets.yeast

#print dynpy.mxutils.raise_matrix_power(b.dyn.trans, 10)
#asdsf
# print b.dyn.trans


b = dynpy.bn.BooleanNetwork(rules=dynpy.sample_nets.yeast_cellcycle_bn)
print b.stgIgraph().components()

#print b.dyn.getAttractorDistribution()
# print b.dyn.equilibriumDistribution()

