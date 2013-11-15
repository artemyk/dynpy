DynPy
=====

DynPy is a framework for building dynamical systems in Python.  So far, it has basic support for linear dynamical systems, random walkers, Boolean Networks, and Cellular Autamata.

For documentation, see the GitHub pages at https://github.iu.edu/pages/akolchin/DynPy/ .  The tutorial, at https://github.iu.edu/pages/akolchin/DynPy/tutorial.html , may be particularly helpful.


Installation
------------
Just download into a directory.  DynPy requires numpy, scipy, matplotlib, and the igraph library http://igraph.sourceforge.net/ .  (note that igraph cannot be installed using pip)


Building the documentation
--------------------------

The documentation is built using Sphinx from the ``docs/`` directory.  To build the HTML documentation locally, use ``make html``. To make the HTML documentation and push it to the GitHub pages repository, do ``./make_gh_pages.sh``.


Running tests
-------------

Run the following from the root directory in order to run tests (including doctests extracted from the documentation):

``nosetests -v --with-doctest``


