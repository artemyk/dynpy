import os
from setuptools import setup

def read(fname):
	c_dir = os.path.dirname(os.path.realpath(__file__))
	return open(c_dir + '/' + fname).read()


exec(read('dynpy/version.py'))

setup(name='dynpy',
      version=__version__,
      description='Dynamical systems for Python',
      author='Artemy Kolchinsky',
      author_email='artemyk@gmail.com',
      url='https://github.com/artemyk/dynpy',
      packages=['dynpy'],
      install_requires=['numpy>=1.6','scipy>=0.13','coverage>=3.7.0','six>=1.8.0'],
      license="GPL",
      long_description=read('README.md'),
	  download_url='https://codeload.github.com/artemyk/dynpy/tar.gz/0.2#egg=dynpy-0.2.tar.gz',
      classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python",
		"Programming Language :: Python :: 3",
    ],
     )
