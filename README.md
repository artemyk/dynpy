DynPy
=====

DynPy is a framework for building dynamical systems in Python.  So far, it has basic support for linear dynamical systems, random walkers, Boolean Networks, and Cellular Autamata.

For documentation, see the GitHub pages at https://github.iu.edu/pages/akolchin/DynPy/ .  


Building the documentation
--------------------------

The documentation is built using Sphinx from the ``docs/`` directory.  From directory, you can run tests by doing:
``make doctest``

To build the HTML documentation locally, use:

``make html``

To make the HTML documentation and push it to the GitHub pages repository, do:

``./make_gh_pages.sh``


