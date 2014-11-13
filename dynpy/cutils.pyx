"""Cython file with utility functions.
"""

import numpy as np
cimport numpy as np

cpdef inline int tuple2int(list tup):
    """Helper function which converts a binary representation (e.g.,
        ``[1,0,1]``) into an integer
    """
    cdef int r = 0
    cdef int i
    for i in tup:
        r <<= 1
        r = r|i
    return r

cpdef int2tuple(val, num_places):
    """Helper function which converts an integer into a binary representation
    (in the form of a numpy array of 0s and 1s). The binary representation will
    be `num_places` long, with extra places padded with 0s.
    """
    cdef np.ndarray[np.uint8_t, ndim=1] r = np.zeros(num_places, np.uint8)
    cdef int i
    for i in range(num_places):
        r[num_places-i-1] = val & 1
        val >>= 1
    return r