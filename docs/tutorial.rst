DynPy tutorial
==============

Introduction
------------
asdfsdsfdsdfadfs
adsfsdfdsf


Random walkers
--------------

For random walker:

.. plot::
	:include-source:

	import matplotlib.pyplot as plt
	import dynpy

	num_steps = 80
	rw = dynpy.graphdynamics.RandomWalker(graph=dynpy.sample_nets.karateclub_net)

	cState = np.zeros(rw.num_vars)
	cState[ 5 ] = 1
	spacetime = np.zeros((num_steps,rw.num_vars))
	for i in range(num_steps):
	    spacetime[i,:] = cState
	    cState = rw.iterate(cState)

	# Also possible, instead of the for loop, to do:
	# spacetime = rw.getTrajectory(startState=cState, last_timepoint=num_steps)	

	plt.spy(spacetime)



Ensembles
---------

For random walker ensemble:

.. plot::
	:include-source:

	import matplotlib.pyplot as plt
	import dynpy

	rw = dynpy.graphdynamics.RandomWalker(graph=dynpy.sample_nets.karateclub_net)
	rwEnsemble = dynpy.dynsys.DynamicalSystemEnsemble(rw)

	initState = np.zeros(rw.num_vars, 'float')
	initState[ 5 ] = 1

	plt.imshow(rwEnsemble.getTrajectory(startState=initState, last_timepoint=80), interpolation='none')	


For continuous-time random walker ensemble:

.. plot::
	:include-source:

	import matplotlib.pyplot as plt
	import dynpy

	rw = dynpy.graphdynamics.RandomWalker(graph=dynpy.sample_nets.karateclub_net, discrete_time = False )
	rwEnsemble = dynpy.dynsys.DynamicalSystemEnsemble(rw)

	initState = np.zeros(rw.num_vars, 'float')
	initState[ 5 ] = 1
	plt.imshow(rwEnsemble.getTrajectory(startState=initState, last_timepoint=80, logscale=True), interpolation='none')	


It is possible to get the equilibrium distribution quickly using eigen decomposition:

.. plot::
	:include-source:

	import matplotlib.pyplot as plt
	import numpy as np
	import dynpy

	rw = dynpy.graphdynamics.RandomWalker(graph=dynpy.sample_nets.karateclub_net, discrete_time = False )
	rwEnsemble = dynpy.dynsys.DynamicalSystemEnsemble(rw)

	plt.imshow(np.atleast_2d(rwEnsemble.equilibriumState()), interpolation='none')	



Boolean Networks
----------------

Let's try to get space time diagram of yeast network


* :class:`dynpy.bn.BooleanNetwork`

.. plot:: 
   :include-source:

	import numpy as np, matplotlib.pyplot as plt
	import dynpy

	bn = dynpy.bn.BooleanNetwork(rules=dynpy.sample_nets.yeast_cellcycle_bn)

	initState = np.zeros(bn.num_vars, 'int')
	initState[ [1,3,6] ] = 1
	plt.spy(bn.getTrajectory(startState=initState, last_timepoint=15))


We can also get its attractors, by doing:

>>> import dynpy
>>> bn = dynpy.bn.BooleanNetwork(rules=dynpy.sample_nets.yeast_cellcycle_bn)
>>> atts, attbasins = bn.getAttractorsAndBasins()
>>> print map(len, attbasins)
[1764, 151, 109, 9, 7, 7, 1]



An ensemble:

.. plot::
	:include-source:

	import matplotlib.pyplot as plt
	import dynpy

	bn = dynpy.bn.BooleanNetwork(rules=dynpy.sample_nets.yeast_cellcycle_bn)
	bnEnsemble = dynpy.dynsys.DynamicalSystemEnsemble(bn)

	# get distribution over states at various timepoints
	t = bnEnsemble.getTrajectory(startState=bnEnsemble.getUniformDistribution(), last_timepoint=20)

	# project back onto original nodes
	bnProbs = t.dot(bn.state2ndxMx)

	# plot
	plt.imshow(bnProbs, interpolation='none')	


Cellular Automata
-----------------

For a CA:

.. plot::
   :include-source:

	import numpy as np, matplotlib.pyplot as plt
	import dynpy

	ca = dynpy.ca.CellularAutomaton(num_vars=100, num_neighbors=1, ca_rule_number=110)

	initState = np.zeros(ca.num_vars, 'int')
	initState[int(ca.num_vars/2)] = 1
	plt.spy(ca.getTrajectory(startState=initState, last_timepoint=50))

