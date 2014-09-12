dynpy [![Build Status](https://travis-ci.org/artemyk/dynpy.svg?branch=master)](https://travis-ci.org/artemyk/dynpy)
=====

dynpy is a framework for building dynamical systems in Python.  So far, it has basic support for linear dynamical systems, random walkers, Boolean Networks, and Cellular Autamata.

For documentation, see the GitHub pages at http://artemyk.github.io/dynpy/html/ .  The tutorial, at http://artemyk.github.io/dynpy/html/tutorial.html , may be particularly helpful.


Installation
------------
Just download into a directory.  dynpy requires numpy, scipy, matplotlib, and the igraph library http://igraph.sourceforge.net/ .  (note that igraph cannot be installed using pip)


Building the documentation
--------------------------

The documentation is built using Sphinx from the ``docs/`` directory.  To build the HTML documentation locally, use ``make html``.

The docstrings should follow NumPy conventions, as specified in https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt .

Running tests
-------------

Run the following from the root directory in order to run tests (including doctests extracted from the documentation):

``nosetests -v --with-doctest``


