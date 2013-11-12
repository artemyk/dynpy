DynPy tutorial
==============

Let's try to get space time diagram of yeast network

.. plot:: 
   :include-source:

	import numpy as np, matplotlib.pyplot as plt
	import dynpy.bn, dynpy.sample_bn_nets

	num_steps = 15
	bn = dynpy.bn.BooleanNetworkFromTruthTables(rules=dynpy.sample_bn_nets.yeast)

	spacetime = np.zeros(shape=(num_steps,bn.num_nodes), dtype='int')

	cState = np.zeros(bn.num_nodes, 'int')
	cState[ [1,3,6] ] = 1

	for i in range(num_steps):
	    spacetime[i,:] = cState
	    cState = bn.iterateState(cState)

	plt.spy(spacetime)


We can also get its attractors, by doing:

>>> import dynpy.bn, dynpy.sample_bn_nets
>>> bn = dynpy.bn.BooleanNetworkFromTruthTables(rules=dynpy.sample_bn_nets.yeast)
>>> atts, attbasins = bn.getAttractorsAndBasins()
>>> print map(len, attbasins)
[1764, 151, 109, 9, 7, 7, 1]



For a CA:

.. plot::
   :include-source:

	import numpy as np, matplotlib.pyplot as plt
	import dynpy.ca

	num_nodes = 100
	num_steps = 50
	ca = dynpy.ca.CellularAutomaton(num_nodes=num_nodes, num_neighbors=1, ca_rule_number=110)
	spacetime = np.zeros(shape=(num_steps,num_nodes), dtype='int')

	cState = np.zeros(num_nodes, 'int')
	cState[int(num_nodes/2)] = 1

	for i in range(num_steps):
	    spacetime[i,:] = cState
	    cState = ca.iterateState(cState)

	plt.spy(spacetime)


For random walker:

.. plot::
	:include-source:

	import networkx as nx, matplotlib.pyplot as plt
	import dynpy.graphdynamics

	num_steps = 80
	rw = dynpy.graphdynamics.RandomWalker(graph=nx.to_numpy_matrix( nx.karate_club_graph() ) )

	spacetime = np.zeros(shape=(num_steps,rw.num_nodes), dtype='int')

	cState = np.zeros(rw.num_nodes, 'int')
	cState[ 5 ] = 1

	for i in range(num_steps):
	    spacetime[i,:] = cState
	    cState = rw.iterateState(cState)

	plt.spy(spacetime)	


For random walker ensemble:

.. plot::
	:include-source:

	import networkx as nx, matplotlib.pyplot as plt
	import dynpy.graphdynamics

	num_steps = 80
	rw = dynpy.graphdynamics.RandomWalker(graph=nx.to_numpy_matrix( nx.karate_club_graph() ), transMatrixClass=dynpy.mx.DenseMatrix )
	rwEnsemble = dynpy.dynsys.DynamicalSystemEnsemble(rw)
	spacetime = np.zeros(shape=(num_steps,rw.num_nodes), dtype='float')

	cState = np.zeros(rw.num_nodes, 'float')
	cState[ 5 ] = 1

	for i in range(num_steps):
	    spacetime[i,:] = cState
	    cState = rwEnsemble.iterateState(cState)

	plt.imshow(spacetime)	
