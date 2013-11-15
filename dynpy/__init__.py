import os
import glob
__all__ = [os.path.basename(f)[:-3] for f in glob.glob(os.path.dirname(__file__) + "/*.py")]
__all__ = [f for f in __all__ if f != '__init__.py']

for module in __all__:
    __import__(module, locals(), globals())
del module
