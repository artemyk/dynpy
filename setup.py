#!/usr/bin/env python

exec(open('dynpy/version.py').read())

from distutils.core import setup

setup(name='dynpy',
      version=__version__,
      description='Dynamical systems for Python',
      author='Artemy Kolchinsky',
      author_email='artemyk@gmail.com',
      url='https://github.com/artemyk/dynpy',
      packages=['dynpy'],
     )