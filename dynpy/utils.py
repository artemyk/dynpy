"""Module which implements various helpful functions
"""
#TODO: Document

from __future__ import division, print_function, absolute_import

import six
range = six.moves.range
map   = six.moves.map

import numpy as np

from . import mx

class readonlydict(dict):
    def __setitem__(self, key, value):
        raise Exception('Read-only dictionary')
    def __delitem__(self, key):
        raise Exception('Read-only dictionary')

def hashable_state(x):
    if not isinstance(x, np.ndarray):
        return x
    else:
        return mx.hashable_array(x)

def is_int(x):
	return isinstance(x, int) or isinstance(x, np.integer)


