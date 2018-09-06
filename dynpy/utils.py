"""Module which implements various helpful functions
"""
#TODO: Document

from __future__ import division, print_function, absolute_import

import six
range = six.moves.range
map   = six.moves.map

import numpy as np

class readonlydict(dict):
    def __setitem__(self, key, value):
        raise Exception('Read-only dictionary')
    def __delitem__(self, key):
        raise Exception('Read-only dictionary')

def is_int(x):
	return isinstance(x, int) or isinstance(x, np.integer)


