dynpy tutorial
==============

Introduction
------------

dynpy is a package for defining and running dynamical systems in Python.  The
goal is to support a wide-variety of dynamical systems, both continuous and
discrete time as well as continuous and discrete state.

dynpy is organized into a hierarchy of classes, with each class representing a
different kind of dynamical system.    The base classes are defined in
:doc:`dynpy.dynsys`.  Some definitions used for creating sample systems in this 
tutorial are defined in :doc:`dynpy.sample_nets`.


Example: Boolean Networks
-------------------------

Boolean network are a type of discrete-state, discrete-time dynamical system.  
Each node updates itself as a Boolean function of other nodes (it's 'inputs').

:doc:`dynpy.bn` implement Boolean network dynamical systems. For example, we
can use it to compute the 
space time diagram of the 11-node yeast cell-cycle network, as described in:
Li et al, The yeast cell-cycle network is robustly designed, PNAS, 2004.


.. plot:: test_pyplots/bn_trajectory.py
   :include-source:


We can also get the network's attractors, by doing:

>>> import dynpy
>>> bn = dynpy.bn.BooleanNetwork(rules=dynpy.sample_nets.budding_yeast_bn)
>>> atts, attbasins = bn.get_attractor_basins(sort=True)
>>> print(list(map(len, attbasins)))
[1764, 151, 109, 9, 7, 7, 1]


Or print them out using:

>>> import dynpy
>>> bn = dynpy.bn.BooleanNetwork(rules=dynpy.sample_nets.budding_yeast_bn)
>>> bn.print_attractor_basins()
* BASIN 1 : 1764 States
ATTRACTORS:
   Cln3    MBF    SBF Cln1,2   Sic1   Swi5  Cdc20 Clb5,6   Cdh1 Clb1,2   Mcm1
      0      0      0      0      1      0      0      0      1      0      0
--------------------------------------------------------------------------------
* BASIN 2 : 151 States
ATTRACTORS:
   Cln3    MBF    SBF Cln1,2   Sic1   Swi5  Cdc20 Clb5,6   Cdh1 Clb1,2   Mcm1
      0      0      1      1      0      0      0      0      0      0      0
--------------------------------------------------------------------------------
* BASIN 3 : 109 States
ATTRACTORS:
   Cln3    MBF    SBF Cln1,2   Sic1   Swi5  Cdc20 Clb5,6   Cdh1 Clb1,2   Mcm1
      0      1      0      0      1      0      0      0      1      0      0
--------------------------------------------------------------------------------
* BASIN 4 : 9 States
ATTRACTORS:
   Cln3    MBF    SBF Cln1,2   Sic1   Swi5  Cdc20 Clb5,6   Cdh1 Clb1,2   Mcm1
      0      0      0      0      1      0      0      0      0      0      0
--------------------------------------------------------------------------------
* BASIN 5 : 7 States
ATTRACTORS:
   Cln3    MBF    SBF Cln1,2   Sic1   Swi5  Cdc20 Clb5,6   Cdh1 Clb1,2   Mcm1
      0      0      0      0      0      0      0      0      0      0      0
--------------------------------------------------------------------------------
* BASIN 6 : 7 States
ATTRACTORS:
   Cln3    MBF    SBF Cln1,2   Sic1   Swi5  Cdc20 Clb5,6   Cdh1 Clb1,2   Mcm1
      0      1      0      0      1      0      0      0      0      0      0
--------------------------------------------------------------------------------
* BASIN 7 : 1 States
ATTRACTORS:
   Cln3    MBF    SBF Cln1,2   Sic1   Swi5  Cdc20 Clb5,6   Cdh1 Clb1,2   Mcm1
      0      0      0      0      0      0      0      0      1      0      0
--------------------------------------------------------------------------------


Cellular Automata
-----------------

The cellular automata class :class:`dynpy.ca.CellularAutomaton` is defined in
:doc:`dynpy.ca`.  It is a subclass of :class:`dynpy.bn.BooleanNetwork`.
It constructs a Boolean network with a lattice connectivity
topology and a homogenous update function.  For example:

.. plot:: test_pyplots/ca_trajectory.py
   :include-source:



Markov Chains
--------------

`dynpy` also implements Markov chains, or dynamical systems over distributions of
states.  See the documentation for :class:`dynpy.markov.MarkovChain` for more 
details. 

For example, here we use :doc:`dynpy.graphdynamics`, which implements dynamics on
graphs, to instantiate a dynamical system representing the distribution
of a random walker on Zachary's karate club network.  Here, 
:class:`dynpy.graphdynamics.RandomWalker` is a subclass of
:class:`dynpy.markov.MarkovChain`.

.. plot:: test_pyplots/mc_trajectory.py
    :include-source:


A Markov chain, like some other dynamical systems implemented by `dynpy`, can also 
be run in continuous time (in this context, it is sometimes called a 'master
equation'). This is specified by passing in the ``discrete_time=False`` option when
constructing the underlying dynamical system. Using the previous example:

.. plot:: test_pyplots/mc_cont_trajectory.py
    :include-source:


It is also possible to get the equilibrium distribution by calling
``get_equilibrium_distribution()``, which uses eigenspace decomposition:

.. plot:: equil_dist.py
    :include-source:


In fact, it is possible to turn any deterministic dynamical system into a Markov 
chain by using the :meth:`dynpy.markov.MarkovChain.from_deterministic_system` method.
For example, to create a dynamical system over a distribution of states of 
the yeast-cell cycle Boolean network:

.. plot:: bn_dist.py
    :include-source:



Stochastic Systems
----------------------------

Stochastic systems can also be implemented.

.. plot::
    :include-source:

>>> import matplotlib.pyplot as plt
>>> import numpy as np
>>> import dynpy
>>> 
>>> num_steps = 30
>>> G = dynpy.sample_nets.karateclub_net
>>> N = G.shape[0]
>>> rw = dynpy.graphdynamics.RandomWalkerEnsemble(graph=G)
>>> sampler = dynpy.markov.MarkovChainSampler(rw)
>>> 
>>> # Initialize with a single random walker on node id=5
>>> cState = np.zeros(N)
>>> cState[ 5 ] = 1
>>> 
>>> spacetime = sampler.get_trajectory(start_state=cState, max_time=num_steps)
>>> 
>>> plt.spy(spacetime)
>>> plt.xlabel('Node')
>>> plt.ylabel('Time')

