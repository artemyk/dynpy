import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext

# http://stackoverflow.com/questions/19919905/how-to-bootstrap-numpy-installation-in-setup-py
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy as np
        self.include_dirs.append(np.get_include())

def read(fname):
  c_dir = os.path.dirname(os.path.realpath(__file__))
  return open(c_dir + '/' + fname).read()

exec(read('dynpy/version.py'))

download_url = "https://github.com/artemyk/dynpy/tarball/master#egg=" + \
               "dynpy-%s.tar.gz" % __version__

cython_modules = ['cutils','bniterate']

try:
    from Cython.Distutils import build_ext
    ext_modules = cythonize(cython_modules)

except ImportError:
    ext_modules = [Extension('dynpy.'+s, ['dynpy/'+s+'.c']) for s in cython_modules]

REQUIRED_NUMPY = 'numpy>=1.6'
required_packages = [
    REQUIRED_NUMPY,
    'scipy>=0.13',
    'six>=1.8.0',
    'coverage>=3.7.0',
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
      setup_requires=[REQUIRED_NUMPY],
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
      cmdclass=dict(build_ext=build_ext),
      ext_modules=ext_modules,
     )
