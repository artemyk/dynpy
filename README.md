dynpy [![Build Status](https://travis-ci.org/artemyk/dynpy.svg?branch=master)](https://travis-ci.org/artemyk/dynpy) [![Documentation Status](https://readthedocs.org/projects/dynpy/badge/)](https://readthedocs.org/projects/dynpy/)
===================================================================================================================

dynpy is a package for defining and running dynamical systems in Python.  The
goal is to support a wide-variety of dynamical systems, both continuous and
discrete time as well as continuous and discrete state.

So far, it has basic support for linear dynamical systems, Markov chains, 
Boolean Networks, and Cellular Autamata.  

It supports both Python2 and Python3.

For documentation, see http://dynpy.readthedocs.org/ .  The tutorial, at 
http://dynpy.readthedocs.org/tutorial.html , may be particularly helpful.


Requirements
------------

dynpy requires numpy, scipy, and six.  
Testing requires nose and coverage.  
Building the documentation requires matplotlib.
