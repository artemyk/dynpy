dynpy tutorial
==============

Introduction
------------

dynpy is a package for defining and running dynamical systems in Python.  The
goal is to support a wide-variety of dynamical systems, both continuous and
discrete time as well as continuous and discrete state, with easy extensibility
and a clean programming interface.

dynpy is organized into a hierarchy of classes, with each class representing a
different kind of dynamical system.    The base classes are defined in
:doc:`dynpy.dynsys`.  Some sample systems used in this tutorial are defined in
:doc:`dynpy.sample_nets`.


Random walkers
--------------

:doc`dynpy.graphdynamics` provides tools to study dynamics on graphs.  Here is a
:simple example of how to initialize and plot several steps of a random walker
:on Zachary's karate club network

.. plot::
    :include-source:

    import matplotlib.pyplot as plt
    import numpy as np
    import dynpy

    num_steps = 30
    rw = dynpy.graphdynamics.RandomWalker(graph=dynpy.sample_nets.karateclub_net)

    # Initialize with a single random walker on node id=5
    cState = np.zeros(rw.num_vars)
    cState[ 5 ] = 1

    spacetime = np.zeros((num_steps,rw.num_vars))
    for i in range(num_steps):
        spacetime[i,:] = cState
        cState = rw.iterate(cState)

    # Also possible, instead of for loop:
    # spacetime = rw.getTrajectory(startState=cState, max_time=num_steps) 

    plt.spy(spacetime)
    plt.xlabel('Node')
    plt.ylabel('Time')


Notice how it's possible to get the spacetime trajectory more succinctly using
the  ``getTrajectory`` method.


Dynamics over state distributions
---------------------------------

The above example of the random walker is a stochastic dynamical system.  It is
also possible to define a dynamical system over the state-distribution of such a
system, which is deterministic linear dynamical system over the space of
distributions.   To do so, we use the state-transition  graph of the underlying
system to generate a Markov chain (or, in the continuous-time case, master
equation) over the states of the underlying system. Each state of the underlying
system is assigned to a separate variable in the Markov chain system; the value
of each variable is the probability mass on the corresponding state of the
underlying system.  See the documentation for :class:`dynpy.dynsys.MarkovChain`
for more details. Using the previous example:

.. plot::
    :include-source:

    import matplotlib.pyplot as plt
    import dynpy

    rw = dynpy.graphdynamics.RandomWalker(graph=dynpy.sample_nets.karateclub_net)
    rwMC = dynpy.dynsys.MarkovChain(rw)

    initState = np.zeros(rw.num_vars)
    initState[ 5 ] = 1

    trajectory = rwMC.getTrajectory(startState=initState, max_time=30)
    plt.imshow(trajectory, interpolation='none')
    plt.xlabel('Node')
    plt.ylabel('Time')


Dynamical systems in dynpy can also be run in continuous-time.  This is usually
implemented only for the 'Markov chain' versions (since then the continuous-time
dynamics reduce to a continuous-time linear dynamical system).   This can be
specified by passing in the ``discrete_time=False`` option when constructing the
underlying dynamical system. Using the previous example:

.. plot::
    :include-source:

    import matplotlib.pyplot as plt
    import dynpy

    kc = dynpy.sample_nets.karateclub_net
    rw = dynpy.graphdynamics.RandomWalker(graph=kc, discrete_time=False)
    rwMC = dynpy.dynsys.MarkovChain(rw)

    initState = np.zeros(rw.num_vars, 'float')
    initState[ 5 ] = 1
    trajectory = rwMC.getTrajectory(startState=initState, max_time=30)
    plt.imshow(trajectory, interpolation='none') 
    plt.xlabel('Node')
    plt.ylabel('Time')

It is also possible to get the equilibrium distribution by calling
``equilibriumState()``, which uses eigenspace decomposition:

.. plot::
    :include-source:

    import matplotlib.pyplot as plt
    import numpy as np
    import dynpy

    kc = dynpy.sample_nets.karateclub_net
    rw = dynpy.graphdynamics.RandomWalker(graph=kc, discrete_time=False)
    rwMC = dynpy.dynsys.MarkovChain(rw)

    eqState = rwMC.equilibriumState()
    plt.imshow(np.atleast_2d(dynpy.mx.todense(eqState)), interpolation='none')



Boolean Networks
----------------

:doc:`dynpy.bn` contains tools to run Boolean network dynamics. Let's try to get
space time diagram of the 11-node yeast cell-cycle network, as described in:
Li et al, The yeast cell-cycle network is robustly designed, PNAS, 2004.
http://www.pnas.org/content/101/14/4781.full.pdf


.. plot:: 
   :include-source:

    import numpy as np, matplotlib.pyplot as plt
    import dynpy

    bn = dynpy.bn.BooleanNetwork(rules=dynpy.sample_nets.yeast_cellcycle_bn)

    initState = np.zeros(bn.num_vars, 'int')
    initState[ [1,3,6] ] = 1
    plt.spy(bn.getTrajectory(startState=initState, max_time=15))
    plt.xlabel('Node')
    plt.ylabel('Time')


We can also get the network's attractors, by doing:

>>> import dynpy
>>> bn = dynpy.bn.BooleanNetwork(rules=dynpy.sample_nets.yeast_cellcycle_bn)
>>> atts, attbasins = bn.getAttractorsAndBasins()
>>> print map(len, attbasins)
[1764, 151, 109, 9, 7, 7, 1]


Or print them out using:

>>> import dynpy
>>> bn = dynpy.bn.BooleanNetwork(rules=dynpy.sample_nets.yeast_cellcycle_bn)
>>> bn.printAttractorsAndBasins()
* BASIN 0 : 1764 States
ATTRACTORS:
   Cln3    MBF    SBF Cln1,2   Sic1   Swi5  Cdc20 Clb5,6   Cdh1 Clb1,2   Mcm1
      0      0      0      0      1      0      0      0      1      0      0
--------------------------------------------------------------------------------
* BASIN 1 : 151 States
ATTRACTORS:
   Cln3    MBF    SBF Cln1,2   Sic1   Swi5  Cdc20 Clb5,6   Cdh1 Clb1,2   Mcm1
      0      0      1      1      0      0      0      0      0      0      0
--------------------------------------------------------------------------------
* BASIN 2 : 109 States
ATTRACTORS:
   Cln3    MBF    SBF Cln1,2   Sic1   Swi5  Cdc20 Clb5,6   Cdh1 Clb1,2   Mcm1
      0      1      0      0      1      0      0      0      1      0      0
--------------------------------------------------------------------------------
* BASIN 3 : 9 States
ATTRACTORS:
   Cln3    MBF    SBF Cln1,2   Sic1   Swi5  Cdc20 Clb5,6   Cdh1 Clb1,2   Mcm1
      0      0      0      0      0      0      0      0      1      0      0
--------------------------------------------------------------------------------
* BASIN 4 : 7 States
ATTRACTORS:
   Cln3    MBF    SBF Cln1,2   Sic1   Swi5  Cdc20 Clb5,6   Cdh1 Clb1,2   Mcm1
      0      0      0      0      0      0      0      0      0      0      0
--------------------------------------------------------------------------------
* BASIN 5 : 7 States
ATTRACTORS:
   Cln3    MBF    SBF Cln1,2   Sic1   Swi5  Cdc20 Clb5,6   Cdh1 Clb1,2   Mcm1
      0      1      0      0      0      0      0      0      1      0      0
--------------------------------------------------------------------------------
* BASIN 6 : 1 States
ATTRACTORS:
   Cln3    MBF    SBF Cln1,2   Sic1   Swi5  Cdc20 Clb5,6   Cdh1 Clb1,2   Mcm1
      0      0      0      0      1      0      0      0      0      0      0
--------------------------------------------------------------------------------



Just to demonstrate, it is possible to turn any dynamical system that provides a
state-transition graph (by subclassing
:class:`dynpy.dynsys.DiscreteStateSystemBase` and implementing a `trans`
property) in a linear system over state distributions.  For example, to create a
dynamical system over a distribution of states of the yeast-cell cycle networks,
we can do the following:

.. plot::
    :include-source:

    import matplotlib.pyplot as plt
    import dynpy

    bn = dynpy.bn.BooleanNetwork(rules=dynpy.sample_nets.yeast_cellcycle_bn)
    bnMC = dynpy.dynsys.MarkovChain(bn)

    # get distribution over states at various timepoints
    t = bnMC.getTrajectory(startState=bnMC.getUniformDistribution(), max_time=20)

    # project back from states onto activations of original nodes
    bnProbs = t.dot(bn.ndx2stateMx)

    # plot
    plt.imshow(bnProbs, interpolation='none') 
    plt.xlabel('Node')
    plt.ylabel('Time')



Cellular Automata
-----------------

The cellular automata class :class:`dynpy.ca.CellularAutomaton` is defined in
:doc:`dynpy.ca`.  It is a subclass of :class:`dynpy.bn.BooleanNetwork`.
Effectively, it constructs a Boolean network with a lattice connectivity
topology and a homogenous update function.  Here is an example of how to use it:

.. plot::
   :include-source:

    import numpy as np, matplotlib.pyplot as plt
    import dynpy

    ca = dynpy.ca.CellularAutomaton(num_vars=100, num_neighbors=1, ca_rule_number=110)

    initState = np.zeros(ca.num_vars, 'int')
    initState[int(ca.num_vars/2)] = 1
    plt.spy(ca.getTrajectory(startState=initState, max_time=50))
    plt.xlabel('Node')
    plt.ylabel('Time')

