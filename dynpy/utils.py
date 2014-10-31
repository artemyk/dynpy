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

def tuple2int(bitlist):
    """Helper function which converts a binary representation (e.g.,
        ``[1,0,1]``) into an integer
    """
    # return int("".join(map(str, map(int, bitlist))), 2)
    out = 0
    for bit in bitlist:
        out = (out << 1) | bit
    return out

def int2tuple(i, num_places):
    """Helper function which converts an integer into a binary representation
    (in the form of a numpy array of 0s and 1s). The binary representation will
    be `num_places` long, with extra places padded with 0s.
    """
    return hashable_state(
        np.array(list(map(int,bin(i)[2:].rjust(num_places, '0'))),'int'))

