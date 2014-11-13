# TODO: Document

import numpy as np
cimport numpy as np

include "cutils.pyx"

def iterate_1step_truthtable(object self, np.ndarray[np.uint8_t, ndim=1] starting_state):
    cdef int i
    cdef int ndx
    cdef list cinput
    cdef list cinputs
    cdef list ttable
    cdef np.ndarray[np.uint8_t, ndim=1] nextstate = starting_state.copy()
    for ndx, (_, inputs, ttable) in enumerate(self.rules):
        cinput = [starting_state[i] for i in inputs]
        nextstate[ndx] = ttable[-1-tuple2int(cinput)]
    return nextstate
        