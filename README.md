dynpy [![Build Status](https://travis-ci.org/artemyk/dynpy.svg?branch=master)](https://travis-ci.org/artemyk/dynpy)
=====

dynpy is a framework for building dynamical systems in Python.  So far, it has basic support for linear dynamical systems, random walkers, Boolean Networks, and Cellular Autamata.

For documentation, see http://dynpy.readthedocs.org/ .  The tutorial, at http://dynpy.readthedocs.org/tutorial.html , may be particularly helpful.


Installation
------------
Just download into a directory.  dynpy requires numpy, scipy, matplotlib, and the igraph library http://igraph.sourceforge.net/ .  (note that igraph cannot be installed using pip)


Running tests
-------------

Run the following from the root directory in order to run tests (including doctests extracted from the documentation):

``nosetests -v --with-doctest``


