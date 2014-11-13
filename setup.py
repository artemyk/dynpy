import os
from setuptools import setup

def read(fname):
  c_dir = os.path.dirname(os.path.realpath(__file__))
  return open(c_dir + '/' + fname).read()

exec(read('dynpy/version.py'))

download_url = "https://github.com/artemyk/dynpy/tarball/master#egg=" + \
               "dynpy-%s.tar.gz" % __version__

class lazy_list(list):
    # See http://stackoverflow.com/questions/11010151/distributing-a-shared-library-and-some-c-code-with-a-cython-extension-module
    # We do this so Cython/numpy gets installed before extensions built
    def __init__(self, callback):
        self._list, self.callback = None, callback
    def c_list(self):
        if self._list is None: self._list = self.callback()
        return self._list
    def __iter__(self):
        for e in self.c_list(): yield e
    def __getitem__(self, ii): 
        return self.c_list()[ii]
    def __getslice__(self, i, j):
        return self.c_list()[i:j]
    def __len__(self): 
        return len(self.c_list())
    def append(self, val): 
        self.c_list().append(val)

def extensions():
    from Cython.Build import cythonize
    return cythonize('dynpy/*.pyx')

def numpy_includes():
    import numpy as np
    return [np.get_include(),]

required_packages = [
    'numpy>=1.6',
    'scipy>=0.13',
    'six>=1.8.0',
    'coverage>=3.7.0',
    'cython>=0.21',
    'sphinx>=1.0.0',
    'sphinxcontrib-autorun',
]
# The last four are for testing

setup(name='dynpy',
      version=__version__,
      description='Dynamical systems for Python',
      author='Artemy Kolchinsky',
      author_email='artemyk@gmail.com',
      url='https://github.com/artemyk/dynpy',
      packages=['dynpy'],
      install_requires=required_packages,
      license="GPL",
      long_description=read('README.md'),
      download_url=download_url,
      classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
      ],
      zip_safe=False,
      ext_modules=lazy_list(extensions),
      include_dirs=lazy_list(numpy_includes),
     )
