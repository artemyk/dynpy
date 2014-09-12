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
      install_requires = ['numpy>=1.7','scipy>=0.13','python-igraph>=0.6'],
     )