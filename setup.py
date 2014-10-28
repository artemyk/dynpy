import os
from setuptools import setup

exec(open('dynpy/version.py').read())

setup(name='dynpy',
      version=__version__,
      description='Dynamical systems for Python',
      author='Artemy Kolchinsky',
      author_email='artemyk@gmail.com',
      url='https://github.com/artemyk/dynpy',
      packages=['dynpy'],
      install_requires=['numpy>=1.6','scipy>=0.13','coverage>=3.7.0','six>=1.8.0'],
      license="GPL",
      long_description="""
dynpy is a package for defining and running dynamical systems in Python.  The
goal is to support a wide-variety of dynamical systems, both continuous and
discrete time as well as continuous and discrete state.
""",
	  download_url="https://github.com/artemyk/dynpy/tarball/" + __version__,
      classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python",
		"Programming Language :: Python :: 3",
    ],
     )
