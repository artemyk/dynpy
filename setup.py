import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext

# Needed so numpy is available for include dirs
class build_ext(_build_ext):
    def finalize_options(self):
        super().finalize_options()
        import numpy
        if hasattr(self, "include_dirs") and numpy is not None:
            self.include_dirs.append(numpy.get_include())


cython_modules = ['cutils', 'bniterate']

try:
    from Cython.Build import cythonize
    ext_modules = cythonize([os.path.join('dynpy', s + '.pyx')
                             for s in cython_modules])
except ImportError:
    ext_modules = [Extension('dynpy.' + s, [os.path.join('dynpy', s + '.c')])
                   for s in cython_modules]

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
)
