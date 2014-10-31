"""
This module implements several decorates that can help with caching/memoizing
methods and properties.
"""
from __future__ import division, print_function, absolute_import
import six
range = six.moves.range

from collections import OrderedDict
import functools

class setup_cached_data_method(object):
    """
    Class used to generate memoizing decorators.
    """
    def __init__(self, depends_on_attributes=None, sideeffect_attributes=None,
                 args_dont_use_for_key=None, use_cache_check_func=None):
        self.depends_on_attributes = depends_on_attributes
        self.sideeffect_attributes = sideeffect_attributes
        self.args_dont_use_for_key = args_dont_use_for_key
        self.use_cache_check_func = use_cache_check_func

    def __call__(self, func):
        def wrapper(obj, *args, **kwds):

            kwditems = tuple(kwds.items())
            if self.args_dont_use_for_key is not None:
              kwditems = tuple((k,v) for k, v in kwditems  if k not in self.args_dont_use_for_key)

            dependattrs = None
            if self.depends_on_attributes is not None:
              dependattrs = tuple([getattr(obj, depended_attr) for depended_attr in self.depends_on_attributes])

            try:
                keyEntries = tuple([
                   func,
                   args,
                   kwditems,
                   dependattrs
                   ])
            except TypeError:
                print("Can't hash for tuple:")
                print(" - func=%s\n - args=%s\n - kwds=%s" %
                      (str(func), str(args), str(kwds.items())))
                if self.depends_on_attributes is not None:
                  for depended_attr in self.depends_on_attributes:
                      print(" - attr[%s]=%s" %
                            [depended_attr, getattr(obj, depended_attr)])
                raise

            #key = hash(keyEntries)
            key = keyEntries

            if getattr(obj, '_cache', None) is None:
                obj._cache = OrderedDict()

            if key not in obj._cache or \
              (self.use_cache_check_func is not None and
               not self.use_cache_check_func(obj._cache[key], *args, **kwds)):
                returnValue = func(obj, *args, **kwds)
                sideEffects = None
                if self.sideeffect_attributes is not None:
                  sideEffects = tuple(
                    [getattr(obj, sideeffect, None)
                     for sideeffect in self.sideeffect_attributes])
                obj._cache[key] = (returnValue, sideEffects)
            else:
                returnValue, sideEffects = obj._cache[key]

            if sideEffects is not None:
              for n in range(len(sideEffects)):
                  setattr(obj, self.sideeffect_attributes[n], sideEffects[n])

            return returnValue

        return wrapper


def cached_data_method(f):
    """
    Decorator for caching an object's method
    """
    return functools.wraps(f)( setup_cached_data_method()(f) )


def cached_data_prop(f):
    """
    Decorator for caching an object's property
    """
    return property( functools.wraps(f)( cached_data_method(f) ) )
